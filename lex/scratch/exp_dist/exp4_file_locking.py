import os
import fcntl


file_path = "/lex/scratch/exp_dist/temp.txt"


def evaluate(numbers):
    # Numbers should be n unique natural numbers sorted. Starting from 1 to n with no duplicates.
    n = len(numbers)
    expected_sequence = list(range(1, n + 1))
    return numbers == expected_sequence


class LockedFile:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.file = None

    def __enter__(self):
        # Open the file in 'a+' mode (create if doesn't exist, read, and append)
        self.file = open(self.path, self.mode)

        # Lock the file
        fcntl.flock(self.file, fcntl.LOCK_EX)  # Exclusive lock

        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush the file to ensure all data is written
        self.file.flush()
        os.fsync(self.file.fileno())

        # Unlock the file
        fcntl.flock(self.file, fcntl.LOCK_UN)

        # Close the file
        self.file.close()


def process_file(file):
    # Move to the beginning of the file to read the current content
    file.seek(0)
    content = file.read()

    if content == "":
        file.write("1")
    else:
        numbers = content.split(',')
        numbers = [int(n) for n in numbers]
        assert evaluate(numbers), "That's an error!"

        next_number = len(numbers) + 1
        file.write(f",{next_number}")


def read_and_write_locked():
    for i in range(1000):
        with LockedFile(file_path, 'a+') as file:
            process_file(file)


def read_and_write_default():
    for i in range(1000):
        with open(file_path, "a+") as file:
            process_file(file)


def test_evaluate():
    assert evaluate([1, 2, 3, 4, 5])
    assert not evaluate([0, 1, 2, 3, 4, 5])
    assert not evaluate([0, 1, 2, 3, 5, 4])
    assert not evaluate([1, 2, 3, 4, 4, 5])


if __name__ == '__main__':
    # Use torchrun to launch
    # torchrun --standalone --nproc_per_node=8 exp4_file_locking.py
    # read_and_write_default()
    read_and_write_locked()
