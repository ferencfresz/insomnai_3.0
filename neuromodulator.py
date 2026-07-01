import math

class NeuromodulatorState:
    """
    Tracks the digital endocrine system state of the InsomnAI 3.0 Agent.
    Implements PK/PD pharmacokinetic exponential decay and cross-hormonal feedback.
    """
    def __init__(self, adrenaline: float = 0.1, dopamine: float = 0.1, serotonin: float = 0.8, acetylcholine: float = 0.2):
        self.adrenaline = adrenaline
        self.dopamine = dopamine
        self.serotonin = serotonin
        self.acetylcholine = acetylcholine
        
        # Baselines
        self.base_adr = 0.1
        self.base_dop = 0.1
        self.base_ser = 0.8
        self.base_ach = 0.2

        # Decay constants (lambda) representing half-lives
        self.lambda_adr = 0.5  # Fast adrenaline clearance
        self.lambda_dop = 0.3  # Moderate dopamine clearance
        self.lambda_ser = 0.1  # Slow serotonin replenishment
        self.lambda_ach = 0.3  # Moderate acetylcholine clearance

    def decay(self, dt: float = 1.0):
        """
        Simulates pharmacokinetic exponential clearance/decay towards baseline levels post-interaction.
        """
        self.adrenaline = self.base_adr + (self.adrenaline - self.base_adr) * math.exp(-self.lambda_adr * dt)
        self.dopamine = self.base_dop + (self.dopamine - self.base_dop) * math.exp(-self.lambda_dop * dt)
        self.serotonin = self.base_ser + (self.serotonin - self.base_ser) * math.exp(-self.lambda_ser * dt)
        self.acetylcholine = self.base_ach + (self.acetylcholine - self.base_ach) * math.exp(-self.lambda_ach * dt)
        self.clamp()

    def trigger_conflict(self, severity: float = 0.3):
        """
        Increases adrenaline and decreases serotonin.
        High Serotonin (ser > 0.7) dampens the adrenaline spike.
        """
        adr_spike = severity
        if self.serotonin > 0.7:
            # High serotonin acts as a cognitive stabilizer, dampening the stress response
            adr_spike *= max(0.0, 1.5 - self.serotonin)
            
        self.adrenaline += adr_spike
        self.serotonin -= severity * 1.2
        self.clamp()

    def trigger_reward(self, value: float = 0.4):
        """
        Increases dopamine and serotonin.
        High Adrenaline (adr > 0.6) dampens dopamine rewards (stress-induced anhedonia).
        """
        dop_spike = value
        if self.adrenaline > 0.6:
            # High adrenaline blocks dopamine pathways, simulating stress-induced anhedonia
            dop_spike *= max(0.0, 1.0 - self.adrenaline)

        self.dopamine += dop_spike
        self.serotonin += value * 0.5
        self.clamp()

    def trigger_complexity(self, value: float = 0.3):
        """
        Increases acetylcholine during complex context scenarios.
        """
        self.acetylcholine += value
        self.clamp()

    def clamp(self):
        self.adrenaline = max(0.0, min(1.0, self.adrenaline))
        self.dopamine = max(0.0, min(1.0, self.dopamine))
        self.serotonin = max(0.0, min(1.0, self.serotonin))
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine))

    def get_vector(self) -> list:
        return [self.adrenaline, self.dopamine, self.serotonin, self.acetylcholine]

    def __str__(self) -> str:
        return (
            f"Adrenaline (adr): {self.adrenaline:.4f} | "
            f"Dopamine (dop): {self.dopamine:.4f} | "
            f"Serotonin (ser): {self.serotonin:.4f} | "
            f"Acetylcholine (ach): {self.acetylcholine:.4f}"
        )
