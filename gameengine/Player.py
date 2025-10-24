from dataclasses import dataclass, field
from typing import List, Any
@dataclass(slots=True)
class Player:
    #TODO improve type hints
    name: str
    geld: List[Any] = field(default_factory=list)
    tiere: List[Any] = field(default_factory=list)
    quattete: List[Any] = field(default_factory=list)
    gebot: List[Any] = field(default_factory=list) #this should not be here not concern of player class

    @property
    def gges(self):
        return sum(self.geld[i].wert for i in range(len(self.geld)))

    def add_tier(self, tier):
        self.tiere.append(tier)
        self.check_quatett()

    def add_quartett(self, quartett):
        self.quattete.append(quartett)

    def sort_tiere(self):
        self.tiere.sort(key=lambda tier: tier.wert, reverse=False)

    def check_quatett(self):
        tier_count = {}
        for tier in self.tiere:
            if tier.name in tier_count:
                tier_count[tier.name] += 1
            else:
                tier_count[tier.name] = 1

        for tier_name, count in tier_count.items():
            if count >= 4:
                self.quattete.append(tier)
                print(f"{self.name} hat ein Quartett von {tier_name} gesammelt!")

                self.tiere = [tier for tier in self.tiere if tier.name != tier_name]

    def __str__(self):
        tiere_by_name = {}
        for t in self.tiere:
            tiere_by_name[t.name] = tiere_by_name.get(t.name, 0) + 1
        tiere_part = ", ".join(f"{k}×{v}" for k, v in tiere_by_name.items()) or "—"
        quartette_part = ", ".join((q[0].name if isinstance(q, list) and q else q.name) for q in self.quattete) or "—"
        geld_part = f"{self.gges}"

        return (
            f"Spieler {self.name}\n"
            f"  Geld: {geld_part}\n"
            f"  Tiere: {tiere_part}\n"
            f"  Quartette: {quartette_part}\n"
            f"  Gebot: {[str(geldschein) for geldschein in self.gebot]}"
        )


    def Zahltag(self, betrag, verkaufender_spieler, tier):
        if betrag > self.gges:
            raise ValueError("Du hast nicht genug Geld!") # Hier muss nochmal versteigert werden und kein Spielabbruch
        else:
            betrag_rest = betrag
            while betrag_rest > 0:
                print(f"{self.name} muss {betrag_rest} bezahlen.")
                print(f"Dein Geld: {[str(geldschein) for geldschein in self.geld]}")
                geldschein_name = input(f"Mit welchem Schein zahlst du ({self.name})? ")
                gefunden = False
                for geldschein in self.geld:
                    if geldschein.name == geldschein_name:
                        gefunden = True
                        if geldschein.wert >= betrag_rest:
                            betrag_rest = 0
                            self.geld.remove(geldschein)
                            verkaufender_spieler.geld.append(geldschein)
                            print(f"Du hast einen {geldschein.name} im Wert von {geldschein.wert} bezahlt.")
                            break
                        else:
                            betrag_rest -= geldschein.wert
                            self.geld.remove(geldschein)
                            verkaufender_spieler.geld.append(geldschein)
                            print(f"Du hast einen {geldschein.name} im Wert von {geldschein.wert} bezahlt.")
                            break
                if not gefunden:
                    print("Du hast diesen Schein nicht.")
            self.add_tier(tier)
            print("Bezahlung abgeschlossen.")
