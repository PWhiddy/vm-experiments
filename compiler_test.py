#!/usr/bin/env python3
"""
A tiny C-to-VM compiler using pycparser.

Supported features:
  - int variable declarations and assignments (all variables are stored at fixed memory addresses)
  - Arithmetic expressions: +, -, *, /, and relational operators '<' and '>'
  - Logical operators: && (logical AND) and || (logical OR)
  - if/else statements, for loops (with initialization, condition, and increment)
  - break statements inside loops
  - A special call printf(expr); that prints the number (via the VM’s PRINT instruction)
  - return statements that halt the program

The generated VM instructions use 16–bit words: the upper 8 bits are the opcode and the lower 8 bits the operand.
Memory addresses, immediate values, and jump targets range from 0 to 255.
The opcodes are:
    0x00: LOAD A, immediate     (A = immediate)
    0x01: LOAD B, from memory   (B = memory[operand])
    0x02: ADD B                 (A = A + B; zero_flag set if overflow or result==0)
    0x03: STORE A, to memory    (memory[operand] = A)
    0x04: LOAD A, from memory   (A = memory[operand])
    0x05: JMP                   (PC = operand)
    0x06: JZ                    (if A==0 or zero_flag then PC = operand)
    0x07: PRINT                 (print memory[operand])
    0x08: SUB B                 (A = A - B)
    0x09: MUL B                 (A = A * B, mod 256)
    0x0A: DIV B                 (A = A / B (integer division; if B==0, result=0))
    0x0B: CMP_LT                (if (left operand in B) < (right operand in A) then A=1, else A=0)
    0x0C: CMP_GT                (if (left operand in B) > (right operand in A) then A=1, else A=0)
    0xFF: HALT                  (stop execution)
"""

import sys
import time
from pycparser import c_parser, c_ast

# -----------------------------------------------------------------------------
# The Virtual Machine (using 16-bit instructions)
# -----------------------------------------------------------------------------
class Processor:
    def __init__(self):
        # 256 bytes of memory
        self.memory = [0] * 256
        # Registers: A (accumulator), B (general purpose), PC (program counter)
        self.registers = {'A': 0, 'B': 0, 'PC': 0}
        self.zero_flag = False

    def load_program(self, program):
        # program is a list of 16–bit instructions.
        # We load each 16–bit instruction as two consecutive bytes in memory.
        addr = 0
        for instr in program:
            hi = (instr >> 8) & 0xFF
            lo = instr & 0xFF
            self.memory[addr] = hi
            self.memory[addr+1] = lo
            addr += 2

    def fetch(self):
        # Each instruction is 16 bits (2 bytes).
        addr = self.registers['PC']
        hi = self.memory[addr]
        lo = self.memory[addr+1]
        instr = (hi << 8) | lo
        self.registers['PC'] += 2
        return instr

    def execute(self, instruction):
        opcode = (instruction >> 8) & 0xFF
        operand = instruction & 0xFF

        if opcode == 0x00:  # LOAD A, immediate
            self.registers['A'] = operand
        elif opcode == 0x01:  # LOAD B, from memory
            self.registers['B'] = self.memory[operand]
        elif opcode == 0x02:  # ADD B
            result = self.registers['A'] + self.registers['B']
            self.registers['A'] = result & 0xFF
            self.zero_flag = (result > 0xFF) or (self.registers['A'] == 0)
        elif opcode == 0x03:  # STORE A, to memory
            self.memory[operand] = self.registers['A']
        elif opcode == 0x04:  # LOAD A, from memory
            self.registers['A'] = self.memory[operand]
        elif opcode == 0x05:  # JMP
            self.registers['PC'] = operand * 2  # each instruction is 2 bytes
        elif opcode == 0x06:  # JZ (jump if A==0 or zero_flag is set)
            if self.registers['A'] == 0 or self.zero_flag:
                self.registers['PC'] = operand * 2
        elif opcode == 0x07:  # PRINT
            print(f"Output: {self.memory[operand]}")
        elif opcode == 0x08:  # SUB B
            self.registers['A'] = (self.registers['A'] - self.registers['B']) & 0xFF
        elif opcode == 0x09:  # MUL B
            self.registers['A'] = (self.registers['A'] * self.registers['B']) & 0xFF
        elif opcode == 0x0A:  # DIV B
            if self.registers['B'] != 0:
                self.registers['A'] = self.registers['A'] // self.registers['B']
            else:
                self.registers['A'] = 0
        elif opcode == 0x0B:  # CMP_LT: if B < A then A = 1, else A = 0
            self.registers['A'] = 1 if self.registers['B'] < self.registers['A'] else 0
        elif opcode == 0x0C:  # CMP_GT: if B > A then A = 1, else A = 0
            self.registers['A'] = 1 if self.registers['B'] > self.registers['A'] else 0
        elif opcode == 0xFF:  # HALT
            return False
        return True

    def run(self):
        running = True
        max_iterations = 2000
        iterations = 0
        while running and iterations < max_iterations:
            instr = self.fetch()
            running = self.execute(instr)
            iterations += 1
            #time.sleep(0.016)
        print(f"Ran {iterations} instructions")
        if iterations >= max_iterations:
            print("Program stopped: maximum iterations reached")

# -----------------------------------------------------------------------------
# The Code Generator using pycparser (with logical operators &&, || and break)
# -----------------------------------------------------------------------------
class CodeGenerator(c_ast.NodeVisitor):
    def __init__(self):
        # List of 16–bit instructions.
        self.instructions = []
        # Symbol table: variable name -> memory address.
        # We reserve address 0 for printf output.
        self.symtable = {}
        self.next_addr = 1  # variable addresses start at 1

        # For generating labels and patching jump targets:
        self.labels = {}     # label name -> instruction index
        self.fixups = []     # list of (instr_index, label_name)

        # Reserve fixed temporary addresses:
        # self.temp_addr is used for arithmetic intermediate results.
        self.temp_addr = 250  # choose a high memory location (e.g. 250..253 reserved)
        # For Boolean conversion we reserve additional temporary addresses.
        self.bool_temp1 = 254
        self.bool_temp2 = 253

        # Maintain a stack of break targets for loops.
        self.break_stack = []

    def new_label(self, name_hint="L"):
        label = f"{name_hint}{len(self.labels)}"
        return label

    def mark_label(self, label):
        self.labels[label] = len(self.instructions)

    def emit(self, opcode, operand):
        """Emit a 16–bit instruction with an 8–bit opcode and an 8–bit operand."""
        instr = ((opcode & 0xFF) << 8) | (operand & 0xFF)
        self.instructions.append(instr)

    def emit_jump(self, opcode, label):
        """Emit a jump instruction (JMP or JZ) with a placeholder to be patched later."""
        pos = len(self.instructions)
        self.emit(opcode, 0)  # placeholder operand
        self.fixups.append((pos, label))

    def patch_fixups(self):
        for pos, label in self.fixups:
            if label not in self.labels:
                raise RuntimeError(f"Undefined label: {label}")
            target = self.labels[label]
            self.instructions[pos] = (self.instructions[pos] & 0xFF00) | (target & 0xFF)

    def emit_convert_to_bool(self):
        """
        Emit instructions to convert the value in register A to Boolean (0 or 1).
        The sequence is:
            STORE A into temp (using self.temp_addr)
            LOAD A from that temp
            JZ L_false  (if A == 0, jump to L_false)
            LOAD A, 1   (otherwise, load 1)
            JMP L_end
          L_false:
            LOAD A, 0
          L_end:
        """
        label_false = self.new_label("bool_false")
        label_end = self.new_label("bool_end")
        # Save current A into temp_addr.
        self.emit(0x03, self.temp_addr)    # STORE A, temp_addr
        # Reload A from temp_addr.
        self.emit(0x04, self.temp_addr)    # LOAD A, temp_addr
        # If A==0, jump to label_false.
        self.emit_jump(0x06, label_false)  # JZ to bool_false
        # Otherwise, load 1.
        self.emit(0x00, 1)                 # LOAD A, immediate 1
        self.emit_jump(0x05, label_end)     # JMP to end
        self.mark_label(label_false)
        self.emit(0x00, 0)                 # LOAD A, immediate 0
        self.mark_label(label_end)

    # -------------------------
    # Visitors for top–level nodes
    # -------------------------
    def visit_FileAST(self, node):
        for ext in node.ext:
            self.visit(ext)
        # At the end, if no HALT was emitted by a return, emit HALT.
        self.emit(0xFF, 0)
        self.patch_fixups()

    def visit_FuncDef(self, node):
        # We only support a main() function.
        self.visit(node.body)

    def visit_Compound(self, node):
        for stmt in node.block_items or []:
            self.visit(stmt)

    # -------------------------
    # Statements
    # -------------------------
    def visit_Decl(self, node):
        # Only support int declarations.
        if not (hasattr(node.type, 'type') and node.type.type.names == ['int']):
            raise RuntimeError("Only int type is supported")
        var_name = node.name
        if var_name in self.symtable:
            raise RuntimeError(f"Variable {var_name} already declared")
        self.symtable[var_name] = self.next_addr
        self.next_addr += 1
        if node.init is not None:
            self.visit(node.init)  # result in A
            addr = self.symtable[var_name]
            self.emit(0x03, addr)  # STORE A, addr

    def visit_Assignment(self, node):
        if node.op != "=":
            raise RuntimeError("Only simple assignment is supported")
        self.visit(node.rvalue)
        if not isinstance(node.lvalue, c_ast.ID):
            raise RuntimeError("Only simple variables can be assigned to")
        var_name = node.lvalue.name
        if var_name not in self.symtable:
            raise RuntimeError(f"Undeclared variable {var_name}")
        addr = self.symtable[var_name]
        self.emit(0x03, addr)  # STORE A, addr

    def visit_FuncCall(self, node):
        # Only support printf(expression)
        if not isinstance(node.name, c_ast.ID) or node.name.name != "printf":
            raise RuntimeError("Only printf function is supported")
        if not node.args or not node.args.exprs:
            raise RuntimeError("printf expects one argument")
        self.visit(node.args.exprs[0])
        self.emit(0x03, 0)  # STORE A into address 0 (printing area)
        self.emit(0x07, 0)  # PRINT from address 0

    def visit_If(self, node):
        self.visit(node.cond)
        else_label = self.new_label("else")
        self.emit_jump(0x06, else_label)  # JZ if condition false
        self.visit(node.iftrue)
        if node.iffalse:
            end_label = self.new_label("ifend")
            self.emit_jump(0x05, end_label)  # JMP over else clause
            self.mark_label(else_label)
            self.visit(node.iffalse)
            self.mark_label(end_label)
        else:
            self.mark_label(else_label)

    def visit_For(self, node):
        if node.init is not None:
            self.visit(node.init)
        loop_label = self.new_label("forcond")
        self.mark_label(loop_label)
        if node.cond is not None:
            self.visit(node.cond)
        else:
            self.emit(0x00, 1)  # LOAD A, 1 (always true)
        end_label = self.new_label("forend")
        # Push the break target for this loop.
        self.break_stack.append(end_label)
        self.emit_jump(0x06, end_label)  # Exit loop if condition false.
        self.visit(node.stmt)
        if node.next is not None:
            self.visit(node.next)
        self.emit_jump(0x05, loop_label)
        self.mark_label(end_label)
        # Pop the break target.
        self.break_stack.pop()

    def visit_Break(self, node):
        if not self.break_stack:
            raise RuntimeError("Break statement not within a loop")
        current_break = self.break_stack[-1]
        self.emit_jump(0x05, current_break)

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)
        self.emit(0xFF, 0)  # HALT

    # -------------------------
    # Expressions
    # -------------------------
    def visit_BinaryOp(self, node):
        # Handle logical operators separately.
        if node.op == '&&':
            # Logical AND:
            # Evaluate left operand and convert to Boolean.
            self.visit(node.left)
            self.emit_convert_to_bool()
            self.emit(0x03, self.bool_temp1)
            # Evaluate right operand and convert to Boolean.
            self.visit(node.right)
            self.emit_convert_to_bool()
            self.emit(0x03, self.bool_temp2)
            # Load left Boolean value into A.
            self.emit(0x04, self.bool_temp1)
            # Load right Boolean value into B.
            self.emit(0x01, self.bool_temp2)
            # Multiply: 1*1=1; if either is 0, result is 0.
            self.emit(0x09, 0)  # MUL B
            # Convert the result (if nonzero, 1; else 0).
            self.emit_convert_to_bool()
        elif node.op == '||':
            # Logical OR:
            self.visit(node.left)
            self.emit_convert_to_bool()
            self.emit(0x03, self.bool_temp1)
            self.visit(node.right)
            self.emit_convert_to_bool()
            self.emit(0x03, self.bool_temp2)
            self.emit(0x04, self.bool_temp1)
            self.emit(0x01, self.bool_temp2)
            # Add: if either is 1, sum >= 1.
            self.emit(0x02, 0)  # ADD B
            # Convert the sum to Boolean.
            self.emit_convert_to_bool()
        else:
            # For arithmetic operators and '<' or '>', use the existing scheme.
            self.visit(node.left)
            self.emit(0x03, self.temp_addr)
            self.visit(node.right)
            self.emit(0x01, self.temp_addr)
            if node.op == '+':
                self.emit(0x02, 0)  # ADD B
            elif node.op == '-':
                self.emit(0x08, 0)  # SUB B
            elif node.op == '*':
                self.emit(0x09, 0)  # MUL B
            elif node.op == '/':
                self.emit(0x0A, 0)  # DIV B
            elif node.op == '<':
                self.emit(0x0B, 0)  # CMP_LT
            elif node.op == '>':
                self.emit(0x0C, 0)  # CMP_GT
            else:
                raise RuntimeError(f"Unsupported binary operator {node.op}")

    def visit_ID(self, node):
        var_name = node.name
        if var_name not in self.symtable:
            raise RuntimeError(f"Undeclared variable {var_name}")
        addr = self.symtable[var_name]
        self.emit(0x04, addr)  # LOAD A, from memory

    def visit_Constant(self, node):
        if node.type != "int":
            raise RuntimeError("Only int constants are supported")
        value = int(node.value)
        self.emit(0x00, value & 0xFF)  # LOAD A, immediate

    def visit_UnaryOp(self, node):
        if node.op == '-':
            self.visit(node.expr)
            self.emit(0x00, 0)         # LOAD A, 0
            self.emit(0x03, self.temp_addr)
            self.visit(node.expr)
            self.emit(0x01, self.temp_addr)  # LOAD B, temp
            self.emit(0x08, 0)         # SUB B => 0 - x
        else:
            raise RuntimeError(f"Unsupported unary operator {node.op}")

    def visit_Cast(self, node):
        self.visit(node.expr)

    def generic_visit(self, node):
        for c_name, c in node.children():
            self.visit(c)

# -----------------------------------------------------------------------------
# Main: compile a sample C program and run it on the VM.
# -----------------------------------------------------------------------------
def main():
    source_code = r"""
    int main() {
        int a = 0;
        int b = 1;
        int next = 0;
        
        for (;;) {
            next = a + b;
            printf(next);
            a = b;
            b = next;
            if (next > 128) break;
        }
        return 0;
    }
    """

    print("Compiling source code:")
    print(source_code)
    parser = c_parser.CParser()
    ast = parser.parse(source_code)

    codegen = CodeGenerator()
    codegen.visit(ast)

    program = codegen.instructions
    print("\nGenerated VM program (in hex):")
    print([hex(instr) for instr in program])

    print("\nStarting VM execution...")
    proc = Processor()
    proc.load_program(program)
    proc.run()
    print(f"memory: {proc.memory}")

if __name__ == '__main__':
    main()
