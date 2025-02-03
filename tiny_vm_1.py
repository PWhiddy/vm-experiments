class Processor:
    def __init__(self):
        # 256 bytes of memory
        self.memory = [0] * 256
        # 8-bit registers
        self.registers = {
            'A': 0,  # Accumulator
            'B': 0,  # General purpose
            'PC': 0  # Program counter
        }
        self.zero_flag = False
        
    def load_program(self, program):
        """Load a program into memory"""
        for i, instruction in enumerate(program):
            self.memory[i] = instruction
            
    def fetch(self):
        """Fetch the next instruction"""
        instruction = self.memory[self.registers['PC']]
        self.registers['PC'] += 1
        return instruction
    
    def execute(self, instruction):
        """Execute a single instruction"""
        opcode = instruction >> 4
        operand = instruction & 0x0F
        
        if opcode == 0x0:    # LOAD A, value
            self.registers['A'] = operand
        elif opcode == 0x1:   # LOAD B, [address]
            self.registers['B'] = self.memory[operand]
        elif opcode == 0x2:   # ADD B
            self.zero_flag = False
            result = self.registers['A'] + self.registers['B']
            self.registers['A'] = result & 0xFF  # Keep only 8 bits
            self.zero_flag = (result > 255)  # Set flag if overflow occurred
        elif opcode == 0x3:   # STORE A, address
            self.memory[operand] = self.registers['A']
        elif opcode == 0x4:   # LOAD A, [address]
            self.registers['A'] = self.memory[operand]
        elif opcode == 0x5:   # JMP address
            self.registers['PC'] = operand
        elif opcode == 0x6:   # JZ address
            if self.zero_flag:
                self.registers['PC'] = operand
        elif opcode == 0x7:   # PRINT [address]
            print(f"Output: {self.memory[operand]}")
        elif opcode == 0xF:   # HALT
            return False
        return True
    
    def run(self):
        """Run the program until HALT"""
        running = True
        max_iterations = 300  # Reduced for testing
        iterations = 0
        
        while running and iterations < max_iterations:
            instruction = self.fetch()
            running = self.execute(instruction)
            iterations += 1
            
        if iterations >= max_iterations:
            print("Program stopped: maximum iterations reached")

# Fibonacci program with print instruction
program = [
    0x00,   # 0: LOAD A, 0
    0x30,   # 1: STORE A, 0    ; First number (0)
    0x01,   # 2: LOAD A, 1
    0x31,   # 3: STORE A, 1    ; Second number (1)
    0x71,   # 4: PRINT [1]     ; Print current number
    0x40,   # 5: LOAD A, [0]   ; Load first number
    0x11,   # 6: LOAD B, [1]   ; Load second number from memory into B
    0x20,   # 7: ADD B         ; Add them
    0x6F,   # 8: JZ E          ; If overflow, jump to halt
    0x32,   # 9: STORE A, 2    ; Store sum
    0x41,   # A: LOAD A, [1]   ; Move second number to first position
    0x30,   # B: STORE A, 0
    0x42,   # C: LOAD A, [2]   ; Move sum to second position
    0x31,   # D: STORE A, 1
    0x54,   # E: JMP 4         ; Jump back to print and addition
    0xF0    # F: HALT
]

# Create and run the processor
print("Starting program execution...")
processor = Processor()
processor.load_program(program)
processor.run()
