from bs4 import BeautifulSoup
import json
NUM_FILES = 3999
INSTRUCTIONS = ["pneumonia", "edema", "focal consolidation", "pleural effusion", "pneumothorax"]

"""
Convert medical reports in xml file into json consisting of
image file, target sentence, masked paragraph, and instructions
(keyword).
"""

# Clear output file and prepare output dictionary
outfile = open("captions.json", "w")
output_dict = dict()

# to keep track number of each instruction
instruction_counter = dict()
for instruction in INSTRUCTIONS:
    instruction_counter[instruction] = 0

# Read the XML file and build the dictionary
for i in range (1, NUM_FILES+1):
    temp_dict = dict()

    try:
        with open("NLMCXR_reports/ecgen-radiology/"+str(i)+".xml", "r") as infile:
            contents = infile.read()
            soup = BeautifulSoup(contents,"xml")

            # get caption ---------------------------
            finding = soup.find('AbstractText', Label="FINDINGS").get_text().lower()
            
            if not finding:
                continue # eliminate entries with empty findings
            
            contains_instruction = False
            for instruction in INSTRUCTIONS:
                if instruction in finding:
                    contains_instruction = True
                    temp_dict["instruction"] = instruction
                    break # get the first instruction
            if not contains_instruction:
                continue
             
            finding = finding.split(".")
            target = ""
            for j in range(len(finding)):
                if temp_dict["instruction"] in finding[j]:
                    target = finding[j]
                    finding[j] = "<fill-in-the-blank>"
                    break # get the first sentence mentioning the instruction
            temp_dict["target"] = target

            # reconstruct masked sentence
            finding.pop() # remove the empty string
            masked = ""
            for sentence in finding:
                masked += sentence + " . "
            temp_dict["masked"] = masked
            
            # get image ----------------------------
            image = soup.find('url').get_text()
            if not image:
                continue # eliminate entries with empty images
            image = image.replace("/hadoop/storage/radiology/extract/", "") # remove url
            image = image.replace("jpg", "png") # change filetype
            temp_dict["image"] = image

            # add to output dictionary
            output_dict[i] = temp_dict
            instruction_counter[temp_dict["instruction"]] += 1
    except:
        pass

# write output file
json.dump(output_dict, outfile)
# print("length: " + str(len(output_dict))) # DEBUG
# print("instruction counter: " + str(instruction_counter)) # DEBUG


# Close output file
outfile.close()