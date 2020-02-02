from random import choice

animals = [item.strip() for item in
           """alligator, anteater, armadillo, auroch, axolotl, badger, bat, bear, beaver, blobfish, buffalo, camel,
           chameleon, cheetah, chipmunk, chinchilla, chupacabra, cormorant, coyote, crow, dingo, dinosaur, dog,
           dolphin, dragon, duck, dumbo octopus, elephant, ferret, fox, frog, giraffe, goose, gopher, grizzly,
           hamster, hedgehog, hippo, hyena, jackal, jackalope, ibex, ifrit, iguana, kangaroo, kiwi, koala, kraken,
           lemur, leopard, liger, lion, llama, manatee, mink, monkey, moose, narwhal, nyan cat, orangutan, otter,
           panda, penguin, platypus, python, pumpkin, quagga, quokka, rabbit, raccoon, rhino, sheep, shrew, skunk,
           slow loris, squirrel, tiger, turtle, unicorn, walrus, wolf, wolverine, wombat""".split(',')]


def get_speaker_name() -> str:
    """
    Return randomized speaker name
    """
    return choice(animals)
