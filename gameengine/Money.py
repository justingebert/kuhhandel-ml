from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Money:
    wert:int
    name:str
