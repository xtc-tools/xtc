import subprocess
from os import path, remove, close
from tempfile import NamedTemporaryFile, mkstemp
import re

def debug(msg:str, silent:bool):
    if not silent:
        print(msg)


class DocUtils:
    def __init__(self, 
        compiler:str="gcc",
    ) -> None:
        self.compiler = [compiler]
        self.file_to_delete = set()

    def create_output_file(self) -> str:
        file_descriptor, file_name = mkstemp()
        self.file_to_delete.add(file_name)
        close(file_descriptor)

        return file_name

    def compile(self,
                path_to_file:str,
                args:str,
                silent:bool=True
               ) -> str:
        """Method that tries to compile the given file.
        If sucessful return the executable file name. If anything fails, returns None"""
        if not path.exists(path_to_file):
            raise Exception("Unable to find file: " + path_to_file)

        output_file = self.create_output_file()
        command     = self.compiler + args.split() + ["-o", output_file, path_to_file]

        debug(f"Compiling : {path_to_file} with {command}", silent)
        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            raise Exception(result.stderr.decode())

        return output_file

    def exec_objdump(self,
                     path_to_file:str,
                     args:str,
                     symbol:str="",
                     silent:bool=True
                    ) -> str:
        """Method that tries to use objdump on a binary file
        If an error occurs, it returns the corresponding error as a string"""

        if not path.exists(path_to_file):
            raise Exception("Unable to find file: " + path_to_file)

        if symbol == "":
            disassemble_option = ["--disassemble"]
        else:
            disassemble_option = ["--disassemble=" + symbol]

        command = ["objdump"] + args.split() + disassemble_option + [path_to_file]

        debug(f"Dumping : {path_to_file} with {command}", silent)
        result = subprocess.run(command, capture_output=True)

        return result.stdout.decode() + result.stderr.decode()


    def copy_c_function(self, c_program:str, function_name:str) -> str:
        """Method search the function_name declaration in c_program and extract it,
        based on the number of curly braces couple it found"""
        matched = re.search(f"([\\w.]+[\\s]+{function_name}\\([^)]*\\){{)", c_program, re.ASCII)
        if matched is None:
            raise Exception("Unable to find function: " + function_name)
        
        start_position = matched.end()
        function_content = matched.groups()[0]

        needed_closing_braces = 1 # start at one because we already got the first "{"

        # Copy all the characters until the number of needed_closing_braces is 0 or we arrive at the end of the file
        for i in range(start_position, len(c_program)):
            function_content += c_program[i]
            if c_program[i] == "{":
                needed_closing_braces += 1

            elif c_program[i] == "}":
                needed_closing_braces -= 1

            if needed_closing_braces == 0: # all the } have been found
                break

        return function_content

    def extract_c_function_from_file(self, 
                                     path_to_file:str, 
                                     function_name:str, 
                                     markdown:bool=False, 
                                     silent:bool=True
                                    ) -> str:
        """Method that extract and return a C function <function_name> from <file>"""
        if not path.exists(path_to_file):
            debug("Unable to find file", silent)
            return ""
        
        function_content = ""

        with open(path_to_file, "r") as f:
            file_content = f.read()
            function_content = self.copy_c_function(file_content, function_name)

        if markdown:
            return self.format_code_markdown("c", function_content)
        else:
            return function_content

    def extract_file_content(self, path_to_file:str, silent:bool=True) -> str:
        """Method that extract and return a C function <function_name> from <file>"""
        if not path.exists(path_to_file):
            raise Exception("Unable to find file")

        with open(path_to_file, "r") as f:
            return f.read()

    def format_code_markdown(self, lang:str, code:str) -> str:
        return f"""```{lang}\n{code}\n```"""

    def exec_mlir_loop_from_string(self, args:str, mlir_code:str) -> str:
        """Method that execute mlir-loop on a mlir_code given in argument"""
        f = NamedTemporaryFile(mode="w+", delete=False) # Use a temporary file to store the code, delete=False because we will need mlir-loop to access it after
        name = f.name
        f.write(mlir_code)
        f.close()

        result = self._mlir_loop(args + " " + name)

        remove(name) # Delete the temporary file
        return result

    def _mlir_loop(self, args:str, silent:bool=True) -> str:
        """Method that execute mlir-loop with the given arguments (Better to use mlir_loop_from_string instead)"""
        if args.find("mlir-loop") != 0:
            command = ["mlir-loop"]
        else:
            command = []

        command += args.split()
        result = subprocess.run(command, capture_output=True)

        return result.stderr.decode() + result.stdout.decode()

    def delete_file(self, file_name:str) -> None:
        if file_name not in self.file_to_delete:
            return

        remove(file_name)
        self.file_to_delete.remove(file_name)


