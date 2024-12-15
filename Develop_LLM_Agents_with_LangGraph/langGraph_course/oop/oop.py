from typing import TypedDict
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=2)


class Robot:
    """This class implements a Robot."""
    population = 0  # class attribute
    def __init__(self, name, year):
        self.name = name
        self.year = year
        Robot.population += 1
    
    def __del__(self):
        print("Robot destroyed!")
    
    def __str__(self):
        my_str = f"My name is {self.name} and my year is {self.year}"
        return my_str
    
    def __add__(self, other):
        year_total = self.year + other.year
        return year_total
    
    def set_energy(self, energy):
        self.energy = energy


class Movie(TypedDict):
    title: str
    year: str
    rating: float

def process_movie(movie: Movie) -> None:
    print(f"Title: {movie['title']}")
    print(f"Year: {movie['year']}")
    print(f"Rating: {movie['rating']}")


class Employee(TypedDict, total=False):
    name: str
    age: int
    department: str


class Address(TypedDict):
    street: str
    city: str
    zip: str


class User(TypedDict):
    username: str
    email: str
    address: Address


if __name__ == "__main__":
    r1 = Robot("R1", 2024)
    r2 = Robot("R2", 2040)
    r3 = Robot("R3", 2015)
    print(r1)
    print(r1.__doc__)
    print(f"Robot name: {r1.name}")
    print(r1.__dict__)
    r1.set_energy(500)
    print(f"Robot energy: {r1.energy}")
    print(r1.__dict__)
    print(f"Robot energy: {getattr(r1, 'energy')}")
    print(getattr(r1, "brand", "N/A"))
    print(f"Robots alive: {Robot.population}")
    print(f"R2 + R3 = {r2 + r3}")

    my_movie: Movie = {
        "title": "Inception",
        "year": 2010,
        "rating": 6.6
    }
    print(f"My movie: {my_movie}")
    print(f"Movie title: {my_movie['title']}")
    process_movie(my_movie)

    my_employee: Employee = {
        "name": "Patrick",
        "age": 34,
        "department": "IT/AI/ML"
    }
    print(f"My employee: {my_employee}")

    my_user: User = {
        "username": "Patrick",
        "email": "patrick@example.com",
        "address": {
            "street": "123 Maint St",
            "city": "NYC",
            "zip": "12345"
        }
    }
    pp.pprint(my_user)
    print(f"Username: {my_user['username']}")
    print(f"User email; {my_user['email']}")
    print(f"User Address: {my_user['address']}")