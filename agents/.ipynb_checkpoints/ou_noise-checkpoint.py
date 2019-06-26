{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import copy\n",
    "\n",
    "class OUNoise:\n",
    "    \"\"\"Ornstein-Uhlenbeck process.\"\"\"\n",
    "\n",
    "    def __init__(self, size, mu, theta, sigma):\n",
    "        \"\"\"Initialize parameters and noise process.\"\"\"\n",
    "        self.mu = mu * np.ones(size)\n",
    "        self.theta = theta\n",
    "        self.sigma = sigma\n",
    "        self.reset()\n",
    "\n",
    "    def reset(self):\n",
    "        \"\"\"Reset the internal state (= noise) to mean (mu).\"\"\"\n",
    "        self.state = copy.copy(self.mu)\n",
    "\n",
    "    def sample(self):\n",
    "        \"\"\"Update internal state and return it as a noise sample.\"\"\"\n",
    "        x = self.state\n",
    "        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))\n",
    "        self.state = x + dx\n",
    "        return self.state"
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
