import unittest
import sys
import importlib

class TestDependencies(unittest.TestCase):
    
    def test_python_environment(self):
        self.assertTrue(sys.version_info.major >= 3)

    def test_library_imports(self):
        libs_to_check = [
            'torch', 
            'torchaudio',
            'numpy',
            'numba',
            'vector_quantize_pytorch',
            'transformers',
            'vocos',
            'av',
            'pydub'
        ]

        missing_libs = []
        
        for lib in libs_to_check:
            try:
                importlib.import_module(lib)
            except ImportError:
                missing_libs.append(lib)
        
        if missing_libs:
            self.fail(f"Failed to import: {missing_libs}")

    def test_torch_cpu_processing(self):
        import torch
        x = torch.ones(2, 2)
        y = x + x
        self.assertEqual(y[0][0].item(), 2.0)

if __name__ == '__main__':
    unittest.main()
