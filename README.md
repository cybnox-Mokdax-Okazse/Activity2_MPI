# Evaluating Fundamental SAT Solvers: A Performance Study of Resolution, DP, and DPLL

## Project Overview

This project provides Python implementations of three fundamental Boolean Satisfiability (SAT) solving algorithms:
1.  The Resolution Algorithm
2.  The Davis-Putnam (DP) Algorithm
3.  The Davis-Putnam-Logemann-Loveland (DPLL) Algorithm (with several basic decision heuristics)

These algorithms are experimentally compared by applying them to Sudoku puzzles of varying sizes (4x4 and 9x9) and difficulties. The Sudoku puzzles are first encoded into Conjunctive Normal Form (CNF), which is then passed to the SAT solvers.

The primary goal of this project is to offer a clear, hands-on understanding of these foundational SAT techniques, their theoretical underpinnings, their practical performance characteristics (execution time, memory usage, operational counts), and their inherent trade-offs.

This work was developed as part of the MPI(L)2025 course requirements at the Department of Computer Science, West University of Timișoara.

**Author:** [Your Full Name]
**Email:** `[your.email]@e-uvt.ro`
**Paper Link:** [Placeholder for link to your final paper, e.g., Overleaf link if shareable, or a note like "Paper submitted for MPI(L)2025"]

## Files in this Repository

* `sat_solver_comparison.py`: The main Python script containing the Sudoku-to-CNF encoder, implementations of the SAT solvers, and the experimental execution framework.
* `sudoku_puzzles/` (suggested directory): This directory should contain the input text files with Sudoku puzzles. Example files used in the study might include:
    * `set1_puzzles.txt`
    * `set2_puzzles.txt`
    * (or a single comprehensive file like `all_benchmarks.txt`)
* `README.md`: This file.
* `(Optional)` `LICENSE`: If you choose to add a license.
* `(Optional)` `Your_Paper_MPI_L_2025.pdf`: A PDF version of your final paper.

## Setup and Requirements

* **Python Version:** Python 3.11.9 was used for development. Other Python 3.x versions (e.g., 3.7+) should generally work.
* **Standard Libraries:** The script primarily uses standard Python libraries:
    * `time` (for timing execution)
    * `random` (for the random DPLL heuristic)
    * `tracemalloc` (for memory profiling)
    * `collections` (specifically `defaultdict` and `Counter`)
    * `copy` (for `deepcopy`)
    * `os` (for file operations like checking if `sudoku_puzzles.txt` exists)
    No external libraries need to be installed beyond a standard Python installation.

## How to Run the Experiments

1.  **Prepare Input Puzzles:**
    * Ensure you have a text file containing Sudoku puzzles (e.g., named `sudoku_puzzles.txt` by default, or you can modify the `puzzle_file_to_run` variable in the script).
    * The format for each puzzle in this file must be:
        * A title line starting with `>` (e.g., `>My_Puzzle_Name_1`)
        * Followed by `N` lines, each containing `N` characters representing the Sudoku grid.
        * For an $N \times N$ grid (where $N$ is 4 or 9):
            * Use digits `1` through `N` for pre-filled cells.
            * Use `.` (period) or `0` (zero) for empty cells.
            * Ensure each grid line has exactly `N` characters with no extra spaces.
        * Puzzles can be listed one after another, optionally separated by blank lines.

2.  **Execute the Script:**
    * Open a terminal or command prompt.
    * Navigate to the directory where you have saved `sat_solver_comparison.py` and your puzzle file(s).
    * Run the script using:
        ```bash
        python sat_solver_comparison.py
        ```
    * If your main puzzle input file has a different name than `sudoku_puzzles.txt`, modify this line at the beginning of the `if __name__ == "__main__":` block in the script:
        `puzzle_file_to_run = "your_actual_puzzle_file.txt"`

## Understanding the Output

The script will output the following for each puzzle it processes from the input file:

1.  **Puzzle Information:**
    * The puzzle name and its size (e.g., ` Solving: 4x4_A_VeryEasy (4x4) `).
    * The initial grid.
2.  **CNF Encoding Details:**
    * Time taken for encoding.
    * Number of variables and initial clauses generated.
3.  **For each SAT Solver Applied:**
    * Solver name (and heuristic for DPLL).
    * **Status:**
        * `SAT`: A satisfying assignment was found.
        * `UNSAT`: The formula was proven unsatisfiable.
        * `TIMEOUT`: The solver exceeded its predefined time limit.
        * `MEM_LIMIT`: The solver exceeded its predefined clause generation limit (for Resolution and DP).
        * `CRASH`: An unexpected error occurred during the solver's execution.
    * **Details:** A message from the solver (e.g., "DPLL found a satisfying assignment," "Resolution timed out...").
    * **Time Taken:** Execution time for that solver on that puzzle in seconds.
    * **Peak Memory:** Peak memory usage by Python objects during the solver's run in MB.
    * **Auxiliary Metric:**
        * For Resolution and DP: "Max Clauses Generated/Reached".
        * For DPLL: "DPLL Decisions".
    * **Solution Found:** If status is `SAT` and the solver can produce a model (primarily DPLL, and DP in some cases), the solved Sudoku grid will be printed.

## Notes on Current Implementation

* The SAT solvers are pedagogical implementations intended to demonstrate the core logic of fundamental algorithms. They do not include many of the advanced optimizations found in industrial SAT solvers.
* As discussed in the accompanying paper, the current DP and DPLL implementations may incorrectly report some known satisfiable 9x9 Sudoku instances as UNSAT, typically due to issues in the initial unit propagation phase or specific sensitivities in the CNF encoding used. The case of `9x9_NCS_Z_Very_Sparse_Hard_SAT_Set5` also shows heuristic-dependent correctness for DPLL. These are noted as areas for further investigation.
