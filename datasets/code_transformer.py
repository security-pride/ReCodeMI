#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import io
import json
import keyword
import random
import tokenize
from argparse import Namespace
from contextlib import contextmanager
from typing import Dict, List

import obfupy.document as obfupy_document
import obfupy.documentmanager as obfupy_documentmanager
import obfupy.transformers.formatter as formatter
import obfupy.transformers.rewriter as rewriter
from obfupy.transformers.internal import astutil as obfupy_astutil
from obfupy.transformers.internal import callbackdata as obfupy_callbackdata
from obfupy.transformers.internal import util as obfupy_internal_util
from obfupy.transformers.internal.rewriter import constantmanager as obfupy_constantmanager
from obfupy.transformers.internal.rewriter import context as obfupy_context
from obfupy.transformers.internal.rewriter import functionrewriter as obfupy_functionrewriter


_STABLE_RENAME_SALT = "obfupy"
_STABLE_RENAME_TOKEN_LENGTH = 8
_RESERVED_NAMES = set(keyword.kwlist).union(dir(builtins))


class _MemoryDocument(obfupy_document.Document):
    """Document adapter for the unmodified PyPI package's file-only API."""

    def load_from_content(self, content: str, file_name: str) -> "_MemoryDocument":
        self.setContent(content)
        self._fileName = file_name
        return self


class _CompatibleFormatter:
    """Formatter with the bounded whitespace ranges used by the prior fork."""

    def __init__(self, options=None, callback=None):
        self._options = copy.deepcopy(options if options is not None else formatter.Options())
        self._callback = callback

    def transform(self, document_manager) -> None:
        for document in document_manager.getDocumentList():
            callback_options = obfupy_callbackdata._OptionCallbackData(
                document.getFileName(), self._options
            )
            options = (
                obfupy_callbackdata._invokeCallback(self._callback, callback_options)
                or self._options
            )
            if not options.enabled:
                continue

            tokens = [
                (token_type, token_value)
                for token_type, token_value, _, _, _ in tokenize.tokenize(
                    io.BytesIO(document.getContent().encode("utf-8")).readline
                )
            ]
            self._format_tokens(tokens, options)
            document.setContent(tokenize.untokenize(tokens).decode("utf-8"))

    def _format_tokens(self, tokens, options) -> None:
        min_indent_length = 0
        can_expand_indent = True
        previous_type = None

        for index, (token_type, token_value) in enumerate(tokens):
            if token_type == tokenize.OP and options.addExtraSpaces and token_value != "!":
                spaces = self._random_spaces()
                token_value += spaces
                if previous_type not in {
                    tokenize.ENCODING,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.NL,
                }:
                    token_value = spaces + token_value
                tokens[index] = (token_type, token_value)

            if token_type == tokenize.NEWLINE and options.addExtraNewLines:
                tokens[index] = (token_type, token_value * random.randint(1, 5))

            if token_type == tokenize.COMMENT and options.removeComment:
                tokens[index] = (token_type, "")

            if token_type == tokenize.INDENT:
                if min_indent_length == 0 or len(token_value) < min_indent_length:
                    min_indent_length = len(token_value)
                if min_indent_length > 0 and len(token_value) % min_indent_length != 0:
                    can_expand_indent = False

            previous_type = token_type

        if options.expandIndent and can_expand_indent and min_indent_length > 0:
            new_indent = self._random_spaces()
            for index, (token_type, token_value) in enumerate(tokens):
                if token_type == tokenize.INDENT:
                    tokens[index] = (
                        token_type,
                        new_indent * (len(token_value) // min_indent_length),
                    )

    @staticmethod
    def _random_spaces() -> str:
        if random.randint(0, 1) == 0:
            return "\t" * random.randint(4, 8)
        return " " * random.randint(8, 16)


def _to_base36(number: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if number == 0:
        return "0"
    result = ""
    while number > 0:
        number, index = divmod(number, len(alphabet))
        result = alphabet[index] + result
    return result


def _make_suffix(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    result = ""
    while True:
        index, remainder = divmod(index, len(alphabet))
        result = alphabet[remainder] + result
        if index == 0:
            return result
        index -= 1


def _build_stable_name(
    original_name: str | None,
    file_name: str | None,
    scope_path: str | None,
) -> str:
    payload = (
        f"{_STABLE_RENAME_SALT}\x1f{file_name or ''}\x1f"
        f"{scope_path or ''}\x1f{original_name or ''}"
    )
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=16).digest()
    token = _to_base36(int.from_bytes(digest, byteorder="big"))
    token = (token + ("0" * _STABLE_RENAME_TOKEN_LENGTH))[:_STABLE_RENAME_TOKEN_LENGTH]
    return f"_r_{token}"


def _is_reserved_name(name: str) -> bool:
    return name in _RESERVED_NAMES or obfupy_internal_util.isSpecialFunctionName(name)


def _stable_unused_symbol(
    usedMap=None,
    originalName=None,
    fileName=None,
    scopePath=None,
) -> str:
    if usedMap is None:
        usedMap = obfupy_internal_util._uniqueSymbolMap
    stable_original_name = originalName or "__generated_symbol__"
    base_name = _build_stable_name(stable_original_name, fileName, scopePath)
    if base_name.isidentifier() and not _is_reserved_name(base_name) and base_name not in usedMap:
        usedMap[base_name] = True
        obfupy_internal_util._uniqueSymbolMap[base_name] = True
        return base_name

    index = 0
    while True:
        new_name = f"{base_name}_{_make_suffix(index)}"
        if new_name.isidentifier() and not _is_reserved_name(new_name) and new_name not in usedMap:
            usedMap[new_name] = True
            obfupy_internal_util._uniqueSymbolMap[new_name] = True
            return new_name
        index += 1


def _scope_metadata(current_context) -> tuple[str | None, str | None]:
    names = []
    module_name = None
    context = current_context
    while context is not None:
        if context.isModule():
            module_name = context.getContextName()
            break
        names.append(context.getContextName())
        context = context.getParent()
    names.reverse()
    return module_name, ".".join(names) if names else None


def _compatible_create_new_name(current_context, name=None) -> str:
    file_name, scope_path = _scope_metadata(current_context)
    return _stable_unused_symbol(
        originalName=name,
        fileName=file_name,
        scopePath=scope_path,
    )


def _compatible_extract_function(original_method, instance, node):
    """Give extracted functions the same scope-aware stable name as the prior fork."""
    if not instance._visitor._getOptions().extractFunction:
        return original_method(instance, node)
    if not instance._canExtractFunction(node):
        return original_method(instance, node)

    current_context = instance._visitor.getCurrentContext()
    parent_context = current_context.getParent()
    naming_context = current_context if parent_context is None else parent_context
    file_name, scope_path = _scope_metadata(naming_context)
    first_name = _stable_unused_symbol(
        originalName=node.name,
        fileName=file_name,
        scopePath=scope_path,
    )
    original_symbol_function = obfupy_internal_util.getUnusedRandomSymbol
    first_call = True

    def extract_symbol(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            return first_name
        return _stable_unused_symbol(*args, **kwargs)

    obfupy_internal_util.getUnusedRandomSymbol = extract_symbol
    try:
        return original_method(instance, node)
    finally:
        obfupy_internal_util.getUnusedRandomSymbol = original_symbol_function


@contextmanager
def _obfupy_compatibility():
    """Apply the former fork's behavior without modifying the installed dependency."""
    rename_mixin = obfupy_context._RenameMixin
    function_class = obfupy_functionrewriter.FunctionRewriter
    constant_class = obfupy_constantmanager.ConstantManager
    original_symbol_function = obfupy_internal_util.getUnusedRandomSymbol
    original_create_name = rename_mixin.createNewName
    original_extract_function = function_class._doExtractFunction
    original_make_constant = constant_class._doMakeConstantNode

    obfupy_internal_util.getUnusedRandomSymbol = _stable_unused_symbol
    rename_mixin.createNewName = _compatible_create_new_name
    function_class._doExtractFunction = lambda instance, node: _compatible_extract_function(
        original_extract_function, instance, node
    )
    constant_class._doMakeConstantNode = (
        lambda instance, value: obfupy_astutil.makeConstant(value)
    )
    try:
        yield
    finally:
        obfupy_internal_util.getUnusedRandomSymbol = original_symbol_function
        rename_mixin.createNewName = original_create_name
        function_class._doExtractFunction = original_extract_function
        constant_class._doMakeConstantNode = original_make_constant


def _transform_code_string(content: str, transformers, file_name: str) -> str:
    manager = obfupy_documentmanager.DocumentManager()
    manager.addDocument(_MemoryDocument().load_from_content(content, file_name))
    with _obfupy_compatibility():
        for transformer in transformers:
            if transformer is not None:
                transformer.transform(manager)
    return manager.getDocumentList()[0].getContent()


def build_ns(**kw) -> Namespace:
    base = dict(
        enable_rewriter=True,
        enable_formatter=False,
        rw_rename_local_variable=False,
        rw_alias_function_argument=False,
        rw_extract_function=False,
        rw_extract_constant=False,
        rw_extract_builtin_function=False,
        rw_add_nop_control_flow=False,
        rw_invert_bool_operator=False,
        rw_invert_compare_operator=False,
        rw_wrap_inverted_compare_operator=False,
        rw_expand_if_condition=False,
        rw_rewrite_if=False,
        rw_remove_docstring=False,
        fmt_remove_comment=False,
        fmt_expand_indent=False,
        fmt_add_extra_spaces=False,
        fmt_add_extra_new_lines=False,
    )
    base.update(kw)
    return Namespace(**base)


def build_rewriter(args: Namespace, preserved_names: List[str]) -> rewriter.Options | None:
    if not args.enable_rewriter:
        return None

    opt = rewriter.Options()
    opt.enabled = True

    opt.renameLocalVariable = args.rw_rename_local_variable
    opt.aliasFunctionArgument = args.rw_alias_function_argument
    opt.extractFunction = args.rw_extract_function
    opt.extractConstant = args.rw_extract_constant
    opt.extractBuiltinFunction = args.rw_extract_builtin_function
    opt.addNopControlFlow = args.rw_add_nop_control_flow
    opt.invertBoolOperator = args.rw_invert_bool_operator
    opt.rewriteIf = args.rw_rewrite_if
    opt.removeDocString = args.rw_remove_docstring

    if hasattr(opt, "expandIfCondition"):
        opt.expandIfCondition = args.rw_expand_if_condition

    if hasattr(opt, "invertCompareOperator"):
        if hasattr(opt.invertCompareOperator, "enabled"):
            opt.invertCompareOperator.enabled = args.rw_invert_compare_operator
        if hasattr(opt.invertCompareOperator, "wrapInvertedCompareOperator"):
            opt.invertCompareOperator.wrapInvertedCompareOperator = (
                args.rw_wrap_inverted_compare_operator
            )

    if hasattr(opt, "stringEncoders"):
        opt.stringEncoders = []

    opt.preservedNames = preserved_names
    return opt


def build_formatter(args: Namespace) -> _CompatibleFormatter | None:
    if not args.enable_formatter:
        return None

    opt = formatter.Options()
    opt.enabled = True
    opt.removeComment = args.fmt_remove_comment
    opt.expandIndent = args.fmt_expand_indent
    opt.addExtraSpaces = args.fmt_add_extra_spaces
    opt.addExtraNewLines = args.fmt_add_extra_new_lines
    return _CompatibleFormatter(opt)


def transform_with_obfupy(
    code: str,
    rw_options: rewriter.Options | None,
    fmt: _CompatibleFormatter | None,
) -> str:
    transformers = []
    if rw_options is not None:
        # Rewriter mutates options during transform, so build a fresh instance each call.
        transformers.append(rewriter.Rewriter(rw_options))
    if fmt is not None:
        transformers.append(fmt)
    transformed = _transform_code_string(
        content=code,
        transformers=transformers,
        file_name="<stack2>.py",
    )

    if not isinstance(transformed, str) or not transformed.strip():
        raise ValueError("obfupy produced empty code")
    return transformed


def _ast_fingerprint(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    return ast.dump(tree, include_attributes=False, annotate_fields=True)


def _is_effective_change(
    source_code: str,
    transformed_code: str,
    baseline_unparse_code: str,
    group_name: str,
) -> bool:
    src_fp = _ast_fingerprint(source_code)
    out_fp = _ast_fingerprint(transformed_code)

    changed_vs_baseline = transformed_code != baseline_unparse_code
    ast_changed = src_fp is not None and out_fp is not None and src_fp != out_fp

    # Formatter is intentionally text-level, so compare to baseline unparse output.
    if group_name == "formatter_rewrite":
        return changed_vs_baseline

    # For rewriter-like groups, prefer syntactic (AST) change.
    # If AST parse fails, fall back to textual delta against baseline.
    if src_fp is None or out_fp is None:
        return changed_vs_baseline
    return ast_changed


def build_group_namespaces(groups: List[str]) -> Dict[str, Namespace]:
    known = {
        "operator_rewrite": build_ns(
            rw_invert_bool_operator=True,
            rw_invert_compare_operator=True,
        ),
        "renaming": build_ns(
            rw_rename_local_variable=True,
            rw_alias_function_argument=True,
        ),
        "structure_rewrite": build_ns(
            rw_rewrite_if=True,
            rw_add_nop_control_flow=True,
            rw_extract_function=True,
        ),
        "formatter_rewrite": build_ns(
            enable_rewriter=False,
            enable_formatter=True,
            fmt_expand_indent=True,
            fmt_add_extra_spaces=True,
            fmt_add_extra_new_lines=True,
        ),
    }

    out: Dict[str, Namespace] = {}
    for group in groups:
        if group not in known:
            raise ValueError(f"Unknown group: {group}. Supported: {sorted(known.keys())}")
        out[group] = known[group]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform one code string with one or multiple transform groups."
    )
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Input code string to transform.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="operator_rewrite,renaming,structure_rewrite,formatter_rewrite",
        help="Comma-separated groups to run.",
    )
    parser.add_argument(
        "--preserved-names",
        type=str,
        default="__init__,__name__,__main__",
        help="Comma-separated preserved names for Rewriter.",
    )
    parser.add_argument(
        "--require-orig-compile",
        action="store_true",
        help="Require source code compiles before transform.",
    )
    parser.add_argument(
        "--require-transformed-compile",
        action="store_true",
        help="Require transformed code compiles before output.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print JSON mapping: group -> transformed_code (only successful groups).",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Only output groups whose transformed code differs from source.",
    )
    parser.add_argument(
        "--print-group-header",
        action="store_true",
        help="When not using --json-output, print group headers before each transformed code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_code = args.code
    if not isinstance(source_code, str) or not source_code.strip():
        raise ValueError("--code is empty")

    if args.require_orig_compile:
        compile(source_code, "<source>", "exec")

    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    if not groups:
        raise ValueError("No groups provided via --groups")
    group_ns = build_group_namespaces(groups)

    preserved_names = [x.strip() for x in args.preserved_names.split(",") if x.strip()]

    try:
        baseline_unparse_code = _transform_code_string(
            content=source_code,
            transformers=[],
            file_name="<stack2-baseline>.py",
        )
        if not isinstance(baseline_unparse_code, str):
            baseline_unparse_code = source_code
    except Exception:
        baseline_unparse_code = source_code

    results: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    for group_name, ns in group_ns.items():
        try:
            rw = build_rewriter(ns, preserved_names)
            fmt = build_formatter(ns)
            transformed = transform_with_obfupy(source_code, rw, fmt)
            if args.require_transformed_compile:
                compile(transformed, f"<{group_name}>", "exec")
            if args.only_changed and not _is_effective_change(
                source_code=source_code,
                transformed_code=transformed,
                baseline_unparse_code=baseline_unparse_code,
                group_name=group_name,
            ):
                continue
            results[group_name] = transformed
        except Exception as exc:
            errors[group_name] = f"{type(exc).__name__}: {exc}"

    if args.json_output:
        print(json.dumps({"results": results, "errors": errors}, ensure_ascii=False, indent=2))
        return

    first = True
    for group_name in groups:
        if group_name not in results:
            continue
        if args.print_group_header:
            if not first:
                print()
            print(f"=== {group_name} ===")
        print(results[group_name])
        first = False


if __name__ == "__main__":
    main()
