import msgspec


class Transaction(msgspec.Struct):
    amount: float
    installments: int
    requested_at: str


class Customer(msgspec.Struct):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list[str]


class Merchant(msgspec.Struct):
    id: str
    mcc: str
    avg_amount: float


class Terminal(msgspec.Struct):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransaction(msgspec.Struct):
    timestamp: str
    km_from_current: float


class FraudRequest(msgspec.Struct):
    id: str
    transaction: Transaction
    customer: Customer
    merchant: Merchant
    terminal: Terminal
    last_transaction: LastTransaction | None


class FraudResponse(msgspec.Struct):
    approved: bool
    fraud_score: float
