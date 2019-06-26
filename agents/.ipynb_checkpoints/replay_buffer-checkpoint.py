{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "from collections import namedtuple, deque\n",
    "\n",
    "class ReplayBuffer:\n",
    "    \"\"\"Fixed-size buffer to store experience tuples.\"\"\"\n",
    "\n",
    "    def __init__(self, buffer_size, batch_size):\n",
    "        \"\"\"Initialize a ReplayBuffer object.\n",
    "        Params\n",
    "        ======\n",
    "            buffer_size: maximum size of buffer\n",
    "            batch_size: size of each training batch\n",
    "        \"\"\"\n",
    "        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)\n",
    "        self.batch_size = batch_size\n",
    "        self.experience = namedtuple(\"Experience\", field_names=[\"state\", \"action\", \"reward\", \"next_state\", \"done\"])\n",
    "\n",
    "    def add(self, state, action, reward, next_state, done):\n",
    "        \"\"\"Add a new experience to memory.\"\"\"\n",
    "        e = self.experience(state, action, reward, next_state, done)\n",
    "        self.memory.append(e)\n",
    "\n",
    "    def sample(self, batch_size=64):\n",
    "        \"\"\"Randomly sample a batch of experiences from memory.\"\"\"\n",
    "        return random.sample(self.memory, k=self.batch_size)\n",
    "\n",
    "    def __len__(self):\n",
    "        \"\"\"Return the current size of internal memory.\"\"\"\n",
    "        return len(self.memory)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
