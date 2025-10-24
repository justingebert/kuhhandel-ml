from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Animal:
    wert:int
    name:str