import ast
import unittest
from pathlib import Path


AGENT_PATH = Path(__file__).resolve().parents[1] / 'tdmpc2' / 'tdmpc2.py'


class ExplorationIsolationTest(unittest.TestCase):

	def test_learning_targets_do_not_reference_exploration(self):
		tree = ast.parse(AGENT_PATH.read_text(encoding='utf-8'))
		agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TDMPC2')
		methods = {
			node.name: node
			for node in agent.body
			if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
		}
		for method_name in ('_td_target', '_update', 'update_pi'):
			identifiers = []
			for node in ast.walk(methods[method_name]):
				if isinstance(node, ast.Name):
					identifiers.append(node.id)
				elif isinstance(node, ast.Attribute):
					identifiers.append(node.attr)
			self.assertFalse(any('explore' in name.lower() for name in identifiers))
			self.assertFalse(any('bald' in name.lower() for name in identifiers))


if __name__ == '__main__':
	unittest.main()
