'''
Module for analyzing code snippets to determine the environments, dependencies, and other information needed to run the code.
'''


from enum import StrEnum
from typing import Any, Generator, TypeAlias, TypedDict, Set

import base64

import ast
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_typescript
import sys
import re


class SandboxEnvironment(StrEnum):
    AUTO = 'Auto'

    # Web UI Frameworks
    HTML = 'HTML'
    REACT = 'React'
    VUE = 'Vue'
    GRADIO = 'Gradio'
    STREAMLIT = 'Streamlit'
    PYGAME = 'PyGame'
    MERMAID = 'Mermaid'

    # Runner
    PYTHON_RUNNER = 'Python Runner'
    JAVASCRIPT_RUNNER = 'Javascript Runner'

    # Compiler
    C_RUNNER = 'C Runner'
    CPP_RUNNER = 'C++ Runner'
    # CSHARP_RUNNER = 'C# Runner'
    JAVA_RUNNER = 'Java Runner'
    RUST_RUNNER = 'Rust Runner'
    GOLANG_RUNNER = 'Golang Runner'


def extract_python_imports(code: str) -> list[str]:
    '''
    Extract Python package imports using AST parsing.
    Returns a list of top-level package names.
    '''
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    packages: Set[str] = set()

    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the top-level package name from any dotted path
                    # e.g., 'foo.bar.baz' -> 'foo'
                    if name.name:  # Ensure there's a name
                        packages.add(name.name.split('.')[0])

            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (those starting with dots)
                if node.level == 0 and node.module:
                    # Get the top-level package name
                    # e.g., from foo.bar import baz -> 'foo'
                    packages.add(node.module.split('.')[0])

            # Also check for common dynamic import patterns
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'importlib':
                    # Handle importlib.import_module('package')
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                        packages.add(node.args[0].s.split('.')[0])
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    # Handle __import__('package') and importlib.import_module('package')
                    if node.func.value.id == 'importlib' and node.func.attr == 'import_module':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
                    elif node.func.attr == '__import__':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
        except Exception as e:
            pass
            continue

    # Filter out standard library modules using sys.stdlib_module_names
    std_libs = set(sys.stdlib_module_names)

    return list(packages - std_libs)


def extract_js_imports(code: str) -> list[str]:
    '''
    Extract npm package imports using Tree-sitter for robust parsing.
    Handles both JavaScript and TypeScript code, including Vue SFC.
    Returns a list of package names.
    '''
    try:
        # For Vue SFC, extract the script section first
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()

        # Initialize parsers with language modules
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        js_parser = Parser(Language(tree_sitter_javascript.language()))

        # Try parsing as TypeScript first, then JavaScript
        code_bytes = bytes(code, "utf8")
        try:
            tree = ts_parser.parse(code_bytes)
        except Exception as e:
            pass
            try:
                tree = js_parser.parse(code_bytes)
            except Exception as e:
                pass
                tree = None

        if tree is None:
            raise Exception("Both TypeScript and JavaScript parsing failed")

        packages: Set[str] = set()

        def extract_package_name(node: Node) -> str | None:
            """Extract npm package name from string or template string. 
            Returns None for local aliases like @/ or relative paths."""
            if node.type in ['string', 'string_fragment']:
                pkg_path = code[node.start_byte:node.end_byte].strip('"\'')
                if pkg_path.startswith('.') or pkg_path.startswith('/') or pkg_path.startswith('@/'):
                    return None  # relative, absolute, or alias path

                # Scoped npm package: @scope/package/...
                if pkg_path.startswith('@'):
                    parts = pkg_path.split('/')
                    if len(parts) >= 2:
                        return '/'.join(parts[:2])

                # Regular npm package: "lodash/cloneDeep" -> "lodash"
                return pkg_path.split('/')[0]

            elif node.type == 'template_string':
                content = ''
                has_template_var = False
                for child in node.children:
                    if child.type == 'string_fragment':
                        content += code[child.start_byte:child.end_byte]
                    elif child.type == 'template_substitution':
                        has_template_var = True

                if not content or content.startswith('.') or content.startswith('/') or content.startswith('@/'):
                    return None

                if has_template_var:
                    if content.endswith('-literal'):
                        return 'package-template-literal'
                    return None

                if content.startswith('@'):
                    parts = content.split('/')
                    if len(parts) >= 2:
                        return '/'.join(parts[:2])
                return content.split('/')[0]

            return None

        def visit_node(node: Node) -> None:
            if node.type == 'import_statement':
                # Handle ES6 imports
                string_node = node.child_by_field_name('source')
                if string_node:
                    pkg_name = extract_package_name(string_node)
                    if pkg_name:
                        packages.add(pkg_name)

            elif node.type == 'export_statement':
                # Handle re-exports
                source = node.child_by_field_name('source')
                if source:
                    pkg_name = extract_package_name(source)
                    if pkg_name:
                        packages.add(pkg_name)

            elif node.type == 'call_expression':
                # Handle require calls and dynamic imports
                func_node = node.child_by_field_name('function')
                if func_node and func_node.text:
                    func_name = func_node.text.decode('utf8')
                    if func_name in ['require', 'import']:
                        args = node.child_by_field_name('arguments')
                        if args and args.named_children:
                            arg = args.named_children[0]
                            pkg_name = extract_package_name(arg)
                            if pkg_name:
                                packages.add(pkg_name)

            # Recursively visit children
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return list(packages)

    except Exception as e:
        pass
        # Fallback to basic regex parsing if tree-sitter fails
        packages: Set[str] = set()

        # First try to extract script section for Vue SFC
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()

        # Look for imports
        import_patterns = [
            # dynamic imports
            r'(?:import|require)\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
            # static imports
            r'(?:import|from)\s+[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
            # require statements
            r'require\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
        ]

        for pattern in import_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                pkg_name = match.group(1)
                if not pkg_name.startswith('.'):
                    if pkg_name.startswith('@'):
                        parts = pkg_name.split('/')
                        if len(parts) >= 2:
                            packages.add('/'.join(parts[:2]))
                    else:
                        packages.add(pkg_name.split('/')[0])

        return list(packages)


def determine_python_environment(code: str, install_command: str) -> SandboxEnvironment | None:
    '''
    Determine Python sandbox environment based on install command and AST analysis.
    '''
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for specific framework usage patterns
            if isinstance(node, ast.Name) and node.id == 'gr':
                return SandboxEnvironment.GRADIO
            elif isinstance(node, ast.Name) and node.id == 'st':
                return SandboxEnvironment.STREAMLIT
    except SyntaxError:
        pass

    # Check install command for framework detection
    if install_command and 'pygame' in install_command:
        return SandboxEnvironment.PYGAME
    elif install_command and 'gradio' in install_command:
        return SandboxEnvironment.GRADIO
    elif install_command and 'streamlit' in install_command:
        return SandboxEnvironment.STREAMLIT
    # elif install_command and 'nicegui' in install_command:
    #     return SandboxEnvironment.NICEGUI

    return SandboxEnvironment.PYTHON_RUNNER


def determine_jsts_environment(code: str, install_command: str) -> SandboxEnvironment | None:
    '''
    Determine JavaScript/TypeScript sandbox environment based on install command and AST analysis.
    '''
    # First check for Vue SFC structure
    if '<template>' in code or '<script setup' in code:
        return SandboxEnvironment.VUE

    # Check install command for framework detection
    react_packages = {'react', '@react', 'next', '@next', '@tanstack/react-query', 'react-query'}
    vue_packages = {'vue', '@vue', 'nuxt', '@nuxt'}

    if install_command and any(pkg in install_command for pkg in react_packages):
        return SandboxEnvironment.REACT
    elif install_command and any(pkg in install_command for pkg in vue_packages):
        return SandboxEnvironment.VUE

    try:
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        tree = ts_parser.parse(bytes(code, "utf8"))

        def has_framework_patterns(node: Node) -> tuple[bool, str]:
            # React JSX patterns
            if node.type in ['jsx_element', 'jsx_self_closing_element']:
                return True, 'react'
            # Vue <template> AST element
            elif node.type == 'template_element':
                return True, 'vue'
            # Vue template string with restricted patterns
            elif node.type == 'template_string':
                content = code[node.start_byte:node.end_byte]
                vue_patterns = [
                    r'\bv-(if|else|for|bind|on|model|show|html|text)=',       # Vue directives
                    r'@(?:click|change|input|submit|keyup|keydown)\s*=',      # Event bindings
                    r':(?:class|style|src|href|value|disabled|checked)\s*='   # Attribute bindings
                ]
                if any(re.search(p, content) for p in vue_patterns):
                    return True, 'vue'
            return False, ''

        cursor = tree.walk()

        def visit_node() -> SandboxEnvironment | None:
            is_framework, framework = has_framework_patterns(cursor.node)
            if is_framework:
                return SandboxEnvironment.REACT if framework == 'react' else SandboxEnvironment.VUE
            if cursor.goto_first_child():
                while True:
                    result = visit_node()
                    if result:
                        return result
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()
            return None

        result = visit_node()
        if result:
            return result

        # More targeted Vue detection
        vue_patterns = [
            r'export\s+default\s+{[\s\S]*?(components|props|emits|data|methods|computed|watch)\s*:',
            r'defineComponent\s*\(',
            r'Vue\.extend\s*\(',
            r'createApp\s*\(',
            r'\b(ref|reactive|computed|watch|onMounted|onUnmounted|provide|inject)\s*\(',
            r'defineProps\s*\(',
            r'defineEmits\s*\(',
            r'<[a-zA-Z][^>]+\s+(v-(if|else|for|bind|on|model|show|html|text)|@|:)[^>]*>'  # in tag context
        ]

        for pattern in vue_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return SandboxEnvironment.VUE

    except Exception as e:
        pass

    return SandboxEnvironment.JAVASCRIPT_RUNNER


def detect_js_ts_code_lang(code: str) -> str:
    '''
    Detect whether code is JavaScript or TypeScript using Tree-sitter AST parsing.
    Handles Vue SFC, React, and regular JS/TS files.

    Args:
        code (str): The code to analyze

    Returns:
        str: 'typescript' if TypeScript patterns are found, 'javascript' otherwise
    '''
    # Quick check for explicit TypeScript in Vue SFC
    if '<script lang="ts">' in code or '<script lang="typescript">' in code:
        return 'typescript'

    try:
        # Initialize TypeScript parser
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))

        # Parse the code
        tree = ts_parser.parse(bytes(code, "utf8"))

        def has_typescript_patterns(node: Node) -> bool:
            # Check for TypeScript-specific syntax
            if node.type in {
                'type_annotation',           # Type annotations
                'type_alias_declaration',    # type Foo = ...
                'interface_declaration',     # interface Foo
                'enum_declaration',          # enum Foo
                'implements_clause',         # implements Interface
                'type_parameter',            # Generic type parameters
                'type_assertion',            # Type assertions
                'type_predicate',           # Type predicates in functions
                'type_arguments',           # Generic type arguments
                'readonly_type',            # readonly keyword
                'mapped_type',              # Mapped types
                'conditional_type',         # Conditional types
                'union_type',               # Union types
                'intersection_type',        # Intersection types
                'tuple_type',              # Tuple types
                'optional_parameter',       # Optional parameters
                'decorator',                # Decorators
                'ambient_declaration',      # Ambient declarations
                'declare_statement',        # declare keyword
                'accessibility_modifier',   # private/protected/public
            }:
                return True

            # Check for type annotations in variable declarations
            if node.type == 'variable_declarator':
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True

            # Check for return type annotations in functions
            if node.type in {'function_declaration', 'method_definition', 'arrow_function'}:
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True

            return False

        # Walk the AST to find TypeScript patterns
        cursor = tree.walk()

        def visit_node() -> bool:
            if has_typescript_patterns(cursor.node):
                return True

            # Check children
            if cursor.goto_first_child():
                while True:
                    if visit_node():
                        return True
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()

            return False

        if visit_node():
            return 'typescript'

    except Exception as e:
        pass
        # Fallback to basic checks if parsing fails
        pass

    return 'javascript'


def extract_inline_pip_install_commands(code: str) -> tuple[list[str], str]:
    '''
    Extracts pip install commands from inline code comments and returns both the packages and cleaned code.
    This is useful for cases where pip install commands are written as comments in the code or
    Jupyter notebook-style !pip install commands.

    Args:
        code (str): The code to analyze.

    Returns:
        tuple[list[str], str]: A tuple containing:
            1. List of Python packages extracted from pip install commands in comments
            2. Code with the pip install comments removed
    '''
    python_packages = []
    cleaned_lines = []

    # Regex patterns to match pip install commands in comments and Jupyter-style commands
    pip_patterns = [
        # Comments with pip install
        r'#\s*(?:pip|pip3|python -m pip)\s+install\s+(?:(?:--upgrade|--user|--no-cache-dir|-U)\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)',
        # Jupyter-style !pip install
        r'!\s*(?:pip|pip3|python -m pip)\s+install\s+(?:(?:--upgrade|--user|--no-cache-dir|-U)\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)',
        # Requirements file style pip install
        r'(?:#|!)\s*(?:pip|pip3|python -m pip)\s+install\s+(?:-r\s+[\w\-\.\/]+\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)'
    ]

    # Process each line
    for line in code.splitlines():
        matched = False
        for pattern in pip_patterns:
            match = re.search(pattern, line)
            if match:
                matched = True
                # Extract packages from the command
                pkgs = match.group(1).strip().split()
                # Clean package names (remove version specifiers)
                cleaned_pkgs = [pkg.split('==')[0].split('>=')[0].split('<=')[
                    0].split('~=')[0] for pkg in pkgs]
                python_packages.extend(cleaned_pkgs)

                # Remove the pip install command from the line
                cleaned_line = line[:match.start()].rstrip()
                if cleaned_line:  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)
                break

        if not matched:
            cleaned_lines.append(line)

    # Remove duplicates while preserving order
    python_packages = list(dict.fromkeys(python_packages))

    return python_packages, '\n'.join(cleaned_lines)


def extract_js_from_html_script_tags(code: str) -> list[str]:
    '''
    Extract JavaScript package names from HTML script tags.
    Handles both CDN script tags and inline scripts.

    Args:
        code: HTML code containing script tags

    Returns:
        list[str]: List of package names
    '''
    packages: Set[str] = set()

    # Extract packages from CDN script tags
    script_patterns = [
        # unpkg.com pattern
        r'<script[^>]*src="https?://unpkg\.com/(@?[^@/"]+(?:/[^@/"]+)?(?:@[^/"]+)?)[^"]*"[^>]*>',
        # cdn.jsdelivr.net pattern - explicitly handle /npm/ in the path
        r'<script[^>]*src="https?://cdn\.jsdelivr\.net/npm/(@?[^@/"]+(?:/[^@/"]+)?(?:@[^/"]+)?)[^"]*"[^>]*>',
        # Generic CDN pattern for any domain - exclude common path components
        r'<script[^>]*src="https?://(?!(?:[^"]+/)?(?:npm|dist|lib|build|umd|esm|cjs|min)/)[^"]+?/(@?[\w-]+)(?:/[^"]*)?[^"]*"[^>]*>',
    ]

    seen_packages = set()  # Track packages we've already added to avoid duplicates
    for pattern in script_patterns:
        matches = re.finditer(pattern, code, re.IGNORECASE)
        for match in matches:
            pkg_name = match.group(1)
            if pkg_name.startswith('@'):
                # Handle scoped packages
                parts = pkg_name.split('/')
                if len(parts) >= 2:
                    pkg_name = '/'.join(parts[:2])
            else:
                # Remove version and path components from package name
                pkg_name = pkg_name.split('/')[0].split('@')[0]

            # Skip common path components and duplicates
            if pkg_name and pkg_name not in seen_packages and not pkg_name.lower() in {'npm', 'dist', 'lib', 'build', 'umd', 'esm', 'cjs', 'min'}:
                seen_packages.add(pkg_name)
                packages.add(pkg_name)

    # Extract packages from inline scripts
    script_tags = re.finditer(
        r'<script[^>]*>(.*?)</script>', code, re.DOTALL | re.IGNORECASE)
    for script in script_tags:
        script_content = script.group(1)
        # Check for ES module imports with full URLs
        es_module_patterns = [
            # Match imports from CDN URLs, being careful to extract only the package name
            r'import\s+[\w\s{},*]+\s+from\s+[\'"]https?://[^/]+/npm/([^/@"\s]+)[@/][^"]*[\'"]',
        ]
        found_cdn_import = False
        for pattern in es_module_patterns:
            matches = re.finditer(pattern, script_content)
            for match in matches:
                pkg_name = match.group(1)
                if pkg_name and pkg_name not in seen_packages and not pkg_name.lower() in {'npm', 'dist', 'lib', 'build', 'umd', 'esm', 'cjs', 'min', 'https', 'http'}:
                    seen_packages.add(pkg_name)
                    packages.add(pkg_name)
                    found_cdn_import = True

        # Only check for regular imports if we didn't find a CDN import
        if not found_cdn_import:
            # Remove any URL imports before passing to extract_js_imports
            cleaned_content = re.sub(
                r'import\s+[\w\s{},*]+\s+from\s+[\'"]https?://[^"]+[\'"]', '', script_content)
            packages.update(extract_js_imports(cleaned_content))

    return list(packages)


def extract_code_from_markdown(message: str, enable_auto_env: bool = False) -> tuple[str, str, SandboxEnvironment | None, str] | None:
    '''
    Extracts code from a markdown message by parsing code blocks directly.
    Determines sandbox environment based on code content and frameworks used.

    Returns:
        tuple[str, str, SandboxEnvironment | None, str]: A tuple:
            1. code - the longest code block found
            2. code language
            3. sandbox environment determined from code content
            4. install_command - bash command from ```bash code blocks
    '''
    code_block_regex = r'```(?P<code_lang>[\w\+\#\-\.]*)?[ \t]*\r?\n?(?P<code>.*?)```'
    matches = list(re.finditer(code_block_regex, message, re.DOTALL))

    if not matches:
        return None

    # Define a low-priority list for certain languages
    low_priority_languages = ['bash', 'shell',
                              'sh', 'zsh', 'powershell', 'pwsh', '']

    # Extract bash commands first
    install_command = ""
    bash_matches = [match for match in matches if (match.group('code_lang') or '').lower() in ['bash', 'shell', 'sh']]
    if bash_matches:
        # Use the first bash command found, or concatenate multiple if needed
        install_command = bash_matches[0].group('code').strip()
        if len(bash_matches) > 1:
            # If multiple bash blocks, join them with && or newlines
            install_command = ' && '.join([match.group('code').strip() for match in bash_matches])

    # Find the main code block by avoiding low-priority languages
    main_code = ""
    main_code_lang = ""
    max_length = 0

    for match in matches:
        code = match.group('code').strip()
        code_lang = (match.group('code_lang') or '').lower()
        if code_lang not in low_priority_languages and len(code) > max_length:
            main_code = code
            main_code_lang = code_lang
            max_length = len(code)


    # Define language prefixes for each environment
    python_prefixes = ['py', 'ipython', 'pygame', 'gradio', 'streamlit']
    vue_prefixes = ['vue']
    react_prefixes = ['react', 'next']
    js_prefixes = ['js', 'javascript', 'jsx', 'coffee', 'ecma', 'node', 'es', 'svelte']
    html_prefixes = ['html', 'xhtml', 'htm']
    ts_prefixes = ['ts', 'typescript', 'tsx']
    mermaid_prefixes = ['mermaid', 'mmd']
    c_prefixes = ['c']
    cpp_prefixes = ['cpp', 'c++']
    go_prefixes = ['go', 'golang']
    java_prefixes = ['java']
    rust_prefixes = ['rust']
    csharp_prefixes = ['cs', 'csharp', 'dotnet']

    # Helper function to check if any prefix matches
    def matches_prefix(lang: str, prefixes: list[str]) -> bool:
        return any(lang.lower().startswith(prefix) for prefix in prefixes)

    # Determine sandbox environment based on language
    if matches_prefix(main_code_lang, python_prefixes):
        sandbox_env_name =  determine_python_environment(main_code, install_command)
    elif matches_prefix(main_code_lang, vue_prefixes):
        sandbox_env_name = SandboxEnvironment.VUE
        main_code_lang = detect_js_ts_code_lang(main_code)
    elif matches_prefix(main_code_lang, react_prefixes):
        sandbox_env_name = SandboxEnvironment.REACT
        main_code_lang = detect_js_ts_code_lang(main_code)
    elif ('<!DOCTYPE html>' in main_code and ('<head' in main_code or '<body' in main_code)) or (main_code.strip().startswith('<svg')) or (not matches_prefix(main_code_lang, [*react_prefixes, *vue_prefixes, *js_prefixes, *ts_prefixes]) and ('<html' in main_code or '<!DOCTYPE html>' in main_code)):
        sandbox_env_name = SandboxEnvironment.HTML
        main_code_lang = 'html'
    elif matches_prefix(main_code_lang, js_prefixes):
        main_code_lang = 'javascript'
        sandbox_env_name = determine_jsts_environment(main_code, install_command)
    elif matches_prefix(main_code_lang, ts_prefixes):
        main_code_lang = 'typescript'
        sandbox_env_name = determine_jsts_environment(main_code, install_command)
    elif matches_prefix(main_code_lang, html_prefixes):
        main_code_lang = detect_js_ts_code_lang(main_code)
        sandbox_env_name = SandboxEnvironment.HTML
    elif matches_prefix(main_code_lang, mermaid_prefixes):
        main_code_lang = 'markdown'
        sandbox_env_name = SandboxEnvironment.MERMAID
    elif matches_prefix(main_code_lang, cpp_prefixes):
        main_code_lang = 'cpp'
        sandbox_env_name = SandboxEnvironment.CPP_RUNNER
    elif matches_prefix(main_code_lang, go_prefixes):
        main_code_lang = 'go'
        sandbox_env_name = SandboxEnvironment.GOLANG_RUNNER
    elif matches_prefix(main_code_lang, java_prefixes):
        main_code_lang = 'java'
        sandbox_env_name = SandboxEnvironment.JAVA_RUNNER
    elif matches_prefix(main_code_lang, rust_prefixes):
        main_code_lang = 'rust'
        sandbox_env_name = SandboxEnvironment.RUST_RUNNER
    elif main_code_lang == 'c':
        main_code_lang = 'c'
        sandbox_env_name = SandboxEnvironment.C_RUNNER
    else:
        sandbox_env_name = None

    if not main_code_lang:
        main_code_lang = 'markdown'
    
    return main_code, main_code_lang, sandbox_env_name, install_command


def create_placeholder_svg_data_url(width: int, height: int) -> str:
    '''
    Create a data URL for a placeholder image with given dimensions.
    Uses SVG to create an elegant placeholder.

    Args:
        width: Width of the placeholder image
        height: Height of the placeholder image

    Returns:
        str: Data URL containing the SVG image
    '''
    # Create SVG with gradient background and text
    # Use simpler SVG structure for better browser compatibility
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#f3f4f6"/>
        <text
            x="50%"
            y="50%"
            font-family="Arial, sans-serif"
            font-size="{max(12, min(width, height) // 8)}"
            fill="#6b7280"
            text-anchor="middle"
            dominant-baseline="middle">
            {width} × {height}
        </text>
    </svg>'''

    # Convert to base64 data URL
    try:
        encoded_svg = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        return f'data:image/svg+xml;base64,{encoded_svg}'
    except Exception as e:
        pass
        # Fallback to a simple colored div
        return f'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0ie3dpZHRofSIgaGVpZ2h0PSJ7aGVpZ2h0fSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PC9zdmc+'


def replace_placeholder_urls(code: str) -> str:
    '''
    Replace placeholder image URLs with SVG data URLs.
    Only replaces exact matches of "/api/placeholder/{width}/{height}".

    Args:
        code: The source code containing placeholder URLs

    Returns:
        str: Code with placeholder URLs replaced with data URLs
    '''

    def replacer(match: re.Match) -> str:
        try:
            # Extract width and height from the URL using capturing groups
            width = int(match.group(1))
            height = int(match.group(2))
            
            # Validate dimensions
            if width <= 0 or height <= 0:
                pass
                width, height = 100, 100
            elif width > 10000 or height > 10000:
                pass
                width, height = min(width, 1000), min(height, 1000)
            
            pass
            data_url = create_placeholder_svg_data_url(width, height)
            return data_url
        except Exception as e:
            pass
            # Return a simple fallback
            return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PC9zdmc+'

    # Regular expression pattern to match placeholder URLs
    pattern = r'/api/placeholder/(\d+)/(\d+)'

    try:
        # Replace all occurrences
        result = re.sub(pattern, replacer, code)
        pass
        return result
    except Exception as e:
        pass
        return code  # Return original code if replacement fails


def extract_installation_commands(code: str) -> tuple[list[str], list[str]]:
    '''
    Extracts package installation commands from the code block, preserving version information.

    Args:
        code (str): The code block to analyze.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            1. Python packages from pip install commands (with versions if specified).
            2. npm packages from npm install commands (with versions if specified).
    '''
    python_packages = []
    npm_packages = []

    # Process the code line by line to handle both pip and npm commands
    lines = code.split('\n')
    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Handle pip install commands
        if any(x in line for x in ['pip install', 'pip3 install', 'python -m pip install']):
            # Remove the command part and any flags
            parts = line.split('install', 1)[1].strip()
            # Handle flags at the start
            while parts.startswith(('-', '--')):
                parts = parts.split(None, 1)[1]

            # Split by whitespace, respecting quotes
            current = ''
            in_quotes = False
            quote_char = None
            packages = []

            for char in parts:
                if char in '"\'':
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char.isspace() and not in_quotes:
                    if current:
                        packages.append(current)
                        current = ''
                else:
                    current += char
            if current:
                packages.append(current)

            # Add packages, stripping quotes and ignoring flags
            for pkg in packages:
                pkg = pkg.strip('"\'')
                if pkg and not pkg.startswith(('-', '--')) and not pkg == '-r':
                    python_packages.append(pkg)

        # Handle npm/yarn install commands
        elif any(x in line for x in ['npm install', 'npm i', 'yarn add']):
            # Remove the command part and any flags
            if 'yarn add' in line:
                parts = line.split('add', 1)[1]
            else:
                parts = line.split('install', 1)[
                    1] if 'install' in line else line.split('i', 1)[1]
            parts = parts.strip()

            # Handle flags at the start
            while parts.startswith(('-', '--')):
                parts = parts.split(None, 1)[1] if ' ' in parts else ''

            # Process each package
            for pkg in parts.split():
                if pkg.startswith(('-', '--')) or pkg in ('install', 'i', 'add'):
                    continue

                if pkg.startswith('@'):
                    # Handle scoped packages (e.g., @types/node@16.0.0)
                    if '@' in pkg[1:]:  # Has version
                        pkg_parts = pkg.rsplit('@', 1)
                        base_pkg = pkg_parts[0]  # @scope/name
                        version = pkg_parts[1]  # version
                        npm_packages.append(f"{base_pkg}@{version}")
                    else:
                        npm_packages.append(pkg)
                else:
                    npm_packages.append(pkg)

    # Remove duplicates while preserving order
    python_packages = list(dict.fromkeys(python_packages))
    npm_packages = list(dict.fromkeys(npm_packages))

    # Filter out npm command words
    npm_packages = [p for p in npm_packages if p not in (
        'npm', 'install', 'i', 'add')]

    return python_packages, npm_packages


def validate_dependencies(dependencies: list) -> tuple[bool, str]:
    """
    Validate dependency list format and values.
    Allows empty rows but validates format when package name is specified.
    """
    if not dependencies:
        return True, ""

    valid_types = ["python", "npm"]
    for dep in dependencies:
        # Skip validation for empty rows
        if len(dep) != 3:
            return False, "Each dependency must have type, package and version fields"

        dep_type, pkg_name, version = dep

        # Skip empty rows
        if not pkg_name.strip():
            continue

        if dep_type.lower() not in valid_types:
            return False, f"Invalid dependency type: {dep_type}"

        # Validate version format if specified
        if version.strip():
            if dep_type.lower() == "python":
                # Check for valid pip version specifiers
                if not any(op in version for op in ['==', '>=', '<=', '~=', '>', '<']) and version.lower() != "latest":
                    return False, f"Invalid Python version format for {pkg_name}: {version}"
            elif dep_type.lower() == "npm":
                # Check for valid npm version format (starts with @ or valid semver-like)
                if not (version.startswith('@') or version.lower() == "latest"):
                    return False, f"Invalid NPM version format for {pkg_name}: {version}"

    return True, ""


def extract_java_class_name(java_code: str) -> str:
    '''
    Extract the class name from Java code.
    '''
    match = re.search(r'public\s+class\s+(\w+)', java_code)
    return match.group(1) if match else "Main"