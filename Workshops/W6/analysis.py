# analysis.py (Corrected v2)

import ast

class CodeAnalyzer(ast.NodeVisitor):
    """
    An AST NodeVisitor that correctly builds data and control flow maps
    by passing context during the tree traversal.
    """
    def __init__(self):
        self.data_flow_map = {}
        self.control_flow_map = {}
        self.function_defs = {}
        self.current_control_var = None # Tracks the variable in an active 'if' condition

    def _get_source_from_node(self, node):
        """Helper to get the source taint from a single AST node."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            return self._get_source_from_node(node.left)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
        return None

    def visit_FunctionDef(self, node):
        """Store function definitions to link calls to parameters later."""
        self.function_defs[node.name] = [arg.arg for arg in node.args.args]
        self.generic_visit(node)

    def visit_Assign(self, node):
        """
        Handles all assignments, including tuple unpacking. This is now the single
        source of truth for processing assignments.
        """
        # 1. Get a list of all target variable names (e.g., 'a' or ['a', 'b'])
        target_names = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                target_names.append(t.id)
            elif isinstance(t, ast.Tuple):
                target_names.extend([el.id for el in t.elts if isinstance(el, ast.Name)])

        # 2. Get a list of all source values (e.g., 'x' or ['x', 'y'])
        sources = []
        if isinstance(node.value, ast.Tuple):
            sources.extend([self._get_source_from_node(v) for v in node.value.elts])
        else:
            single_source = self._get_source_from_node(node.value)
            sources = [single_source] * len(target_names) # One source taints all targets

        # 3. Populate the data and control flow maps
        for i, name in enumerate(target_names):
            if i < len(sources) and sources[i] is not None:
                self.data_flow_map.setdefault(name, []).append(sources[i])
                # If we are inside an 'if' block, log the control dependency
                if self.current_control_var:
                    self.control_flow_map[name] = self.current_control_var
    
    def visit_Call(self, node):
        """Links function call arguments to function parameters."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.function_defs:
                params = self.function_defs[func_name]
                for i, arg in enumerate(node.args):
                    if isinstance(arg, ast.Name):
                        self.data_flow_map.setdefault(params[i], []).append(arg.id)

    def visit_If(self, node):
        """
        Manages context. It sets the current control variable before visiting
        child nodes and unsets it after.
        """
        condition_vars = [n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)]
        
        # Store previous context in case of nested ifs
        previous_control_var = self.current_control_var
        if condition_vars:
            self.current_control_var = condition_vars[0]

        # The visitor will now automatically call visit_Assign for any assignments inside
        self.generic_visit(node)

        # Restore context after leaving the if/else block
        self.current_control_var = previous_control_var


class FlowGenerator:
    """
    Generates the execution flow string with corrected logic for tracing dependencies.
    """
    def __init__(self, data_map, control_map):
        self.data_map = data_map
        self.control_map = control_map
        self.data_map.setdefault('data', []).append('res')

    def generate(self, initial_value, final_var, taint_hint):
        """Constructs the full flow string by tracing backwards from a hint."""
        path = []
        is_literal_taint = taint_hint.startswith("'") or taint_hint.startswith('"')
        
        # Case 1: Hybrid flow (e.g., the error string is caused by a variable's value)
        if is_literal_taint and final_var in self.control_map:
            path = [final_var, taint_hint]
            current = self.control_map[final_var] # Start tracing from the control var ('v2')
            path.append(current)
            # Trace the rest of the data flow for the control variable
            while current in self.data_map:
                source = self.data_map[current][-1]
                if source == current: break # Stop if a variable taints itself (e.g., a function parameter)
                path.append(source)
                current = source
        # Case 2: Pure data flow (e.g., res = v1 / v2)
        else:
            path = [final_var]
            current = taint_hint # Start tracing backwards from the hint (e.g., 'v1')
            path.append(current)
            while current in self.data_map:
                source = self.data_map[current][-1]
                if source == current: break
                path.append(source)
                current = source

        path.append(initial_value)
        print("->".join(map(str, reversed(path))))

# --- Main Execution ---
def main():
    with open('calc.py', 'r') as f:
        tree = ast.parse(f.read())

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    flow_gen = FlowGenerator(analyzer.data_flow_map, analyzer.control_flow_map)
    
    print("Demonstrating captured execution flows for calc.py:\n")
    
    # Generate the flow for the successful case: 1000 -> val1 -> v1 -> res
    flow_gen.generate(
        initial_value=1000,
        final_var='res',
        taint_hint='v1'
    )
    
    # Generate the flow for the error case: 0 -> val2 -> v2 -> "Wrong..." -> res
    flow_gen.generate(
        initial_value=0,
        final_var='res',
        taint_hint='"Wrong divisor. Please check input"'
    )

if __name__ == "__main__":
    main()
