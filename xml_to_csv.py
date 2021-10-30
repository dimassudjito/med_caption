from bs4 import BeautifulSoup
NUM_FILES = 3999

"""
Convert medical reports in xml file into csv consisting of
image file and its associated caption to fit the existing 
CNN-LSTM implementation

@Challenges: 
> some files are missing and some FINDINGS labels are missing
    - for now, omit these entries
> images file are in jpg when they should be in png
    - replace the text manually, the filename seem to still correspond
> there are several image for one caption
    - for now, get the first image only
> need to remove a subtring on image name
    - /hadoop/storage/radiology/extract/
"""

# Clear output file and add heading
outfile = open("captions.txt", "w")
outfile.write("image,caption\n")
outfile.close()

# Prepare output file for appending
outfile = open("captions.txt", "a")

# Read the XML file
for i in range (1, NUM_FILES+1):
    try:
        with open("NLMCXR_reports/ecgen-radiology/"+str(i)+".xml", "r") as infile:
            contents = infile.read()
            soup = BeautifulSoup(contents,"xml")

            # get caption
            finding = soup.find('AbstractText', Label="FINDINGS").get_text()
            if not finding:
                continue # eliminate entries with empty findings
            
            # get image
            image = soup.find('url').get_text()
            if not image:
                continue # eliminate entries with empty images
            image = image.replace("/hadoop/storage/radiology/extract/", "") # remove url
            image = image.replace("jpg", "png") # change filetype
            outfile.write(image + "," + finding + "\n")
    except:
        pass

# Close output file
outfile.close()