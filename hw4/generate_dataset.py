import json
from appointment_env import BookingEnv


def main():
    """Генерация датасетов, сохранение в JSON"""

    env = BookingEnv()

    train = env.generate(2000)
    val = env.generate(200)

    with open("train.json", "w") as f:
        json.dump([d.__dict__ for d in train], f, indent=2)

    with open("val.json", "w") as f:
        json.dump([d.__dict__ for d in val], f, indent=2)


if __name__ == "__main__":
    main()
