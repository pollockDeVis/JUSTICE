import xml.etree.ElementTree as ET

tree = ET.parse('../../data/input/inputs_ABM/init_values.xml')
root = tree.getroot()
print('Root is ', root)
for child in root.iter('Class'):
    print(child.attrib)

findings = root.find('Class/[@name="Region"]/Attribute/[@name="opdyn_threshold_close"]')
print(findings.text, findings.attrib['type'])

type_ = findings.attrib['type']
match type_:
    case "float":
        value = float(findings.text)
    case "int":
        value = int(findings.text)
    case _:
     raise Exception(type_+" ::: Is not known as a correct type when reading the values in 'init_values.xml'.")

