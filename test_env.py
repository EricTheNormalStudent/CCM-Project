from env import env

game = env(render_mode="human")
game.reset()

print(game.agent_selection)
print(game.observe(game.agent_selection))

game.step(3)
game.step(3)
game.step(4)
game.step(4)
game.step(5)
game.step(5)
game.step(6)
