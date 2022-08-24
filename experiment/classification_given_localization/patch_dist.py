import re


# construct payload
num = 20
classes = {
    "urls": {"col_prefix": "image", "attr": "url"},
    "bboxes": {"col_prefix": "bbox", "attr": "bbox"}
}
payload = [
    f'<input type="text" class="{class_}" ' + 'value="" ' + value['attr'] + '="${' + value["col_prefix"] + str(i) + '}" />\n' for class_,
    value in classes.items() for i in range(1, num + 1)]

with open("dist/index.html", "r") as f:
    dist = f.read()

with open("dist/index.html", "w") as f:
    f.write(re.sub(r'<div\sid=\"csv\"><\/div>', "".join(payload), dist))
