class Tier:
    def __init__(self, name, wert):
        self.name = name
        self.wert = wert
    def __str__(self):
        return f"{self.name} ({self.wert})"
