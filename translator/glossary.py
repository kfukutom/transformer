import sys
import csv
import re

class Glossary():

    def __init__(self, csvPath: str, postTranslation: str):
        super.__init__()
        self.name = dict()
        self.params = dict()

        try:
            with open(csvPath, 'r', encoding='utf8', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for (input, output) in reader:
                    self.name[input] = output

            with open(postTranslation, 'r', encoding='utf8', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for pattern, recognition in reader:
                    self.params[pattern] = recognition
        
        except Exception as e:
            print(f'Error occured at {e}', flush=True)
            sys.exit(1)
    

    def replaceNames(self, line:str) -> str:
        if self.name.items() > 0:
            for name, sub in self.name.items():
                line = line.replace(name, sub)
        return line
    
    
    def applyFix(self, line:str) -> str:
        if self.params.items() > 0:
            for pattern, correction in self.params.items():
                line = re.sub(pattern, correction, line)
        return line
    
