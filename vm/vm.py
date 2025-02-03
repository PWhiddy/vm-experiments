# -----------------------------------------------------------------------------
# The Virtual Machine (using 16-bit instructions)
# -----------------------------------------------------------------------------
class Processor:
    # 256 bytes of memory default
    def __init__(self, memory_size=256):
        self.memory = [0] * memory_size
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
        elif opcode == 0x0D:
            self.registers['A'] = (self.registers['B'] % self.registers['A'])
        elif opcode == 0x0E:
            self.registers['A'] = 1 if (self.registers['A'] == self.registers['B']) else 0
        elif opcode == 0xFF:  # HALT
            return False
        else:
            raise Exception(f"Unknown opcode: {opcode}")
        return True

    def run(self):
        running = True
        max_iterations = 10000
        iterations = 0
        while running and iterations < max_iterations:
            instr = self.fetch()
            running = self.execute(instr)
            iterations += 1
            #time.sleep(0.016)
        print(f"Ran {iterations} instructions")
        if iterations >= max_iterations:
            print("Program stopped: maximum iterations reached")

def main():
    import argparse
    import struct
    parser = argparse.ArgumentParser(description="Compile C source to a VM binary.")
    parser.add_argument("binary", help="Path to the C source file")
    args = parser.parse_args()

    with open(args.binary, "rb") as f:
        binary_data = f.read()

    program = list(struct.unpack(f"{len(binary_data) // 2}H", binary_data))

    p = Processor()

    p.load_program(program)
    p.run()

if __name__ == '__main__':
    main()