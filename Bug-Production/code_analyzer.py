import ast
from collections import defaultdict


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.metrics = {
            "cbo": 0,  # Coupling Between Objects
            "dit": 0,  # Depth of Inheritance Tree
            "fanIn": defaultdict(int),
            "fanOut": defaultdict(int),
            "lcom": 0,  # Lack of Cohesion of Methods
            "noc": 0,  # Number of Children
            "numberOfAttributes": 0,
            "numberOfAttributesInherited": 0,
            "numberOfLinesOfCode": 0,
            "numberOfMethods": 0,
            "numberOfMethodsInherited": 0,
            "numberOfPrivateAttributes": 0,
            "numberOfPrivateMethods": 0,
            "numberOfPublicAttributes": 0,
            "numberOfPublicMethods": 0,
            "rfc": 0,  # Response for Class
            "wmc": 0,  # Weighted Methods per Class
        }
        self.current_class = None
        self.current_methods = []
        self.attribute_access = defaultdict(set)
        self.inheritance_map = {}
        self.method_calls = defaultdict(set)

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.inheritance_map[node.name] = [
            base.id for base in node.bases if isinstance(base, ast.Name)
        ]

        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        self.metrics["numberOfMethods"] += 1

        if self.current_class:
            self.current_methods.append(node.name)
            if node.name.startswith("__"):
                self.metrics["numberOfPrivateMethods"] += 1
            else:
                self.metrics["numberOfPublicMethods"] += 1

            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                    self.method_calls[node.name].add(stmt.func.attr)

            self.generic_visit(node)

            if self.current_methods:
                self.current_methods.pop()
        else:
            self.metrics["numberOfPublicMethods"] += 1
            self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if self.current_class:
                    attr_name = target.attr
                    if attr_name.startswith("_"):
                        self.metrics["numberOfPrivateAttributes"] += 1
                    else:
                        self.metrics["numberOfPublicAttributes"] += 1
                    self.metrics["numberOfAttributes"] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.current_methods:
            self.attribute_access[self.current_methods[-1]].add(node.attr)
        self.generic_visit(node)

    def visit_Module(self, node):
        self.metrics["numberOfLinesOfCode"] = sum(
            1 for line in ast.unparse(node).splitlines() if line.strip()
        )
        self.generic_visit(node)

    def calculate_metrics(self):
        external_classes = set()
        for calls in self.method_calls.values():
            external_classes.update(calls)

        external_classes -= set(self.inheritance_map.keys())

        self.metrics["cbo"] = len(external_classes)

        self.metrics["dit"] = self._calculate_dit()

        called_methods = set(self.method_calls.keys())

        for method in called_methods:
            self.metrics["fanOut"][method] = len(self.method_calls[method])

        for calls in self.method_calls.values():
            for call in calls:
                self.metrics["fanIn"][call] += 1

        self.metrics["fanIn"].update({method: 0 for method in called_methods if method not in self.metrics["fanIn"]})

        self.metrics["lcom"] = self._calculate_lcom()

        self.metrics["rfc"] = self.metrics["numberOfMethods"] + sum(len(calls) for calls in self.method_calls.values())

        self.metrics["wmc"] = self.metrics["numberOfMethods"]

        child_count = defaultdict(int)
        for parent, children in self.inheritance_map.items():
            for child in children:
                child_count[parent] += 1

        self.metrics["noc"] = sum(child_count.values())

    def _calculate_dit(self):
        def get_depth(cls):
            if cls not in self.inheritance_map or not self.inheritance_map[cls]:
                return 0
            return 1 + max(get_depth(base) for base in self.inheritance_map[cls])

        return max((get_depth(cls) for cls in self.inheritance_map), default=0)

    def _calculate_lcom(self):
        method_list = list(self.attribute_access.keys())
        total_pairs = len(method_list) * (len(method_list) - 1) / 2

        if total_pairs == 0:
            return 1

        shared_attributes = sum(
            len(self.attribute_access[m1] & self.attribute_access[m2])
            for i, m1 in enumerate(method_list)
            for m2 in method_list[i + 1:]
        )

        return 1 - (shared_attributes / total_pairs)

    def analyze(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        self.calculate_metrics()

    def get_metrics(self):
        return self.metrics
