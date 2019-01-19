import os
my_list = os.listdir("./")
import fileinput

for the_file in my_list:
	text_to_search = "\\"
	replacement_text = "/"

	if (".xml" in the_file):
		with fileinput.FileInput(the_file, inplace=True, backup='.bak') as file:
			for line in file:
			    print(line.replace(text_to_search, replacement_text), end='')
