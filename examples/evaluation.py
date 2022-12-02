from examples.blobby import f as blobby
from examples.gearlike import f as gear
from examples.weave import f as weave
from examples.knurling import f as knurling
from examples.pawn import f as pawn
from examples.text import f as text
from examples.image import f as image
from examples.example import f as example





eval_sdfs = [blobby, gear, weave, knurling, pawn, text, image, example]
sdfs_names = ['blobby', 'gear', 'weave', 'knurling', 'pawn', 'text', 'image', 'example']



times = [f.generate(samples=2**8)[1] for f in eval_sdfs]

for name, time in zip(sdfs_names, times):
    print(f'{name: <12} {time: 0.3f}s')