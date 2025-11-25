filename = input("filename: ")
with open(filename, "r") as input:
    with open("gfg_output_file3.py", "w") as output:
        for line in input:
            if "except Exception as e" in line:
                line2 = line.rstrip()+ "# pylint: disable=broad-exception-caught"  # Remove trailing whitespace/newline
                output.write(line2 + "\n")
            else:
                output.write(line)

