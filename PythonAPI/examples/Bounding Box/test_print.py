#episode_loss = []
#episode_number = []

with open("_out/loss_lists.txt", "r") as text_file:
    content = text_file.read()
    file_content = content.splitlines()

    episode_number = [int(i) for i in file_content[0].split()]
    episode_loss = [float(i) for i in file_content[3].split()]

    res = "\n".join("{} {}".format(x, y) for x, y in zip(episode_number, episode_loss))
    print(res)