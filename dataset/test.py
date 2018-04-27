import re

pattern="\\?|？|，|、|\\."

text='今天?'

line = re.subn(pattern,'',text)
print(line)