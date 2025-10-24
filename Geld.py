class Geld:
    def __init__(self,name,wert):
        self.wert=wert
        self.name=name
    def __str__(self):
        return f"{self.name} ({self.wert})"
