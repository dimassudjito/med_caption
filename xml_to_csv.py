from bs4 import BeautifulSoup
NUM_FILES = 3999

findings = []
# Read the XML file
with open("NLMCXR_reports/ecgen-radiology/1.xml", "r") as file:
    contents = file.read()
    soup = BeautifulSoup(contents,"xml")
    finding = soup.find('AbstractText', Label="FINDINGS")
    print(finding.get_text())