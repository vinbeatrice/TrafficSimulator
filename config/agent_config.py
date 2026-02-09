INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
#EPSILON_DECAY = 1200000 #PROVA CON QUESTO
EPSILON_DECAY = 1200000 #con 9000 episodi ha ottenuto 197.0, quindi l'unico errore che ha fatto è stato fermarsi un paio di volte a caso. è arrivato con epsilon a 0.02 al 9000 episodio
#EPSILON_DECAY = 0.0006
DECAY_EPISODES = 7000
N_ACTIONS = 5
N_CHANNELS = 4

# Epsilon value: 0.027328807716233548


# provare ad abbassare epsilon decay e aggiungere scheduler, diminuendo forse anche gli episodi (ex. 1400000 con 9000)