from pycparser import c_parser, c_ast

from compiler import CodeGenerator
from vm import Processor

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run C program on VM")
    parser.add_argument("source_file", help="Path to the C source file")
    args = parser.parse_args()

    with open(args.source_file, "r") as f:
        source_code = f.read()

    parser = c_parser.CParser()
    ast = parser.parse(source_code)

    codegen = CodeGenerator()
    codegen.visit(ast)

    program = codegen.instructions

    p = Processor(memory_size=256)

    p.load_program(program)
    p.run()

if __name__ == '__main__':
    main()