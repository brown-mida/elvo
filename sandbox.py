from generator import Generator

gen = Generator('data-1521428185', '/home/lukezhu/data/ELVOS/elvos_meta_drop1.xls')
print(gen.get_steps_per_epoch())
for each in gen.generate():
    print(each)
