import time
import random
import tracemalloc
from collections import defaultdict, Counter
import copy # For deep copying CNF formulas
import os # For file operations

# # Sudoku Encoder (Conceptually sudoku_encoder.py)

def sudoku_var_to_int(row, col, val, N=9):
    """
    Maps a Sudoku cell-value assignment (row, col, val) to a unique integer variable.
    Assumes 1-indexed row, col, val.
    N is the size of the Sudoku grid (e.g., 9 for 9x9).
    """
    if not (1 <= row <= N and 1 <= col <= N and 1 <= val <= N):
        raise ValueError(f"Sudoku variable indices r={row},c={col},v={val} for N={N} are out of bounds.")
    return (row - 1) * N * N + (col - 1) * N + (val - 1) + 1

def int_to_sudoku_var(var_id, N=9):
    """
    Maps a unique integer variable back to its Sudoku (row, col, val) representation.
    Assumes 1-indexed var_id.
    """
    if not (1 <= var_id <= N * N * N):
        raise ValueError(f"Variable ID {var_id} is out of bounds for Sudoku N={N}.")
    var_id_zero_based = var_id - 1
    val = var_id_zero_based % N + 1
    col = (var_id_zero_based // N) % N + 1
    row = (var_id_zero_based // (N * N)) % N + 1
    return row, col, val

def generate_sudoku_cnf(grid, N=9):
    """
    Generates a CNF formula for a given Sudoku grid.
    'grid' is a list of lists of INTEGERS, where 0 represents an empty cell.
    N is the size (e.g., 9 for 9x9, 4 for 4x4).
    """
    cnf = []
    sqrt_N = int(N**0.5)
    if sqrt_N * sqrt_N != N:
        raise ValueError("N must be a perfect square for box constraints (e.g., 9, 4).")

    # 1. Cell constraints: Each cell must contain at least one value.
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            clause = [sudoku_var_to_int(r, c, v, N) for v in range(1, N + 1)]
            cnf.append(clause)

    # 2. Cell constraints: Each cell contains at most one value.
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            for v1 in range(1, N + 1):
                for v2 in range(v1 + 1, N + 1):
                    cnf.append([-sudoku_var_to_int(r, c, v1, N), -sudoku_var_to_int(r, c, v2, N)])

    # 3. Row constraints:
    for r in range(1, N + 1):
        for v in range(1, N + 1):
            # At least one occurrence of v in row r
            clause = [sudoku_var_to_int(r, c, v, N) for c in range(1, N + 1)]
            cnf.append(clause)
            # At most one occurrence of v in row r
            for c1 in range(1, N + 1):
                for c2 in range(c1 + 1, N + 1):
                    cnf.append([-sudoku_var_to_int(r, c1, v, N), -sudoku_var_to_int(r, c2, v, N)])

    # 4. Column constraints:
    for c in range(1, N + 1):
        for v in range(1, N + 1):
            # At least one occurrence of v in col c
            clause = [sudoku_var_to_int(r, c, v, N) for r in range(1, N + 1)]
            cnf.append(clause)
            # At most one occurrence of v in col c
            for r1 in range(1, N + 1):
                for r2 in range(r1 + 1, N + 1):
                    cnf.append([-sudoku_var_to_int(r1, c, v, N), -sudoku_var_to_int(r2, c, v, N)])

    # 5. Box (subgrid) constraints:
    for box_r_start in range(1, N + 1, sqrt_N):
        for box_c_start in range(1, N + 1, sqrt_N):
            for v in range(1, N + 1):
                box_cells_for_v = []
                for r_offset in range(sqrt_N):
                    for c_offset in range(sqrt_N):
                        r, c_val = box_r_start + r_offset, box_c_start + c_offset # Renamed c to c_val
                        box_cells_for_v.append(sudoku_var_to_int(r, c_val, v, N))
                cnf.append(box_cells_for_v) # At least one
                for i in range(len(box_cells_for_v)): # At most one
                    for j in range(i + 1, len(box_cells_for_v)):
                        cnf.append([-box_cells_for_v[i], -box_cells_for_v[j]])
    
    # 6. Pre-filled clues:
    for r_idx, row_list in enumerate(grid):
        for c_idx, val_in_cell in enumerate(row_list): # Renamed val to val_in_cell
            if val_in_cell != 0: # 0 means empty
                r, c_val = r_idx + 1, c_idx + 1 # Renamed c to c_val
                cnf.append([sudoku_var_to_int(r, c_val, val_in_cell, N)])

    unique_cnf_set = {tuple(sorted(clause)) for clause in cnf}
    return [list(clause) for clause in unique_cnf_set]

def format_sudoku_grid_to_string(grid_solution, N=9):
    """Formats a solved Sudoku grid for printing."""
    if not grid_solution or not any(grid_solution): 
        return "No solution found or model not provided."
    
    sqrt_N = int(N**0.5)
    output_lines = []
    h_border = "+" + ("-------+" * sqrt_N if N==9 else "---+"*sqrt_N) 
    
    for r_idx in range(N):
        if r_idx % sqrt_N == 0: # Add horizontal separator before subgrid rows (also at the top via initial add)
             output_lines.append(h_border)
        
        row_str_simple = []
        for c_idx in range(N):
            if c_idx % sqrt_N == 0: # Add vertical separator before subgrid columns (also at the start)
                row_str_simple.append("|")
            cell_val = grid_solution[r_idx][c_idx]
            row_str_simple.append(f" {cell_val if cell_val != 0 else '.'} ") # Padded for alignment
        row_str_simple.append("|") # Closing vertical border
        output_lines.append("".join(row_str_simple))
    output_lines.append(h_border) # Bottom border

    return "\n".join(output_lines)

def decode_sudoku_solution(model, num_total_vars, N=9):
    if not model: return None
    solved_grid = [[0 for _ in range(N)] for _ in range(N)]
    is_model_dict = isinstance(model, dict)
    for var_id in range(1, num_total_vars + 1):
        is_true = False
        if is_model_dict:
            if model.get(var_id, False): is_true = True
        elif var_id <= len(model) and model[var_id -1]: is_true = True # Assumes list model is 0-indexed
        
        if is_true:
            try:
                r, c, v = int_to_sudoku_var(var_id, N)
                if 1 <= r <= N and 1 <= c <= N: 
                    if solved_grid[r-1][c-1] == 0: solved_grid[r-1][c-1] = v
            except ValueError: pass
    return solved_grid

def load_sudoku_puzzles_from_file(filepath):
    # print(f"DEBUG: Starting to load puzzles from '{filepath}'") # REMOVED
    puzzles = []
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Puzzle file '{filepath}' not found.") # Kept as it's an error, not debug
        return []

    current_grid_lines = []
    current_puzzle_name = None
    N_of_current_grid = 0 
    parsing_a_valid_puzzle_block = False 

    def finalize_puzzle(name_to_finalize, lines_to_finalize, N_for_block, block_was_valid_flag, context_msg):
        # print(f"DEBUG ({context_msg}): Attempting to finalize puzzle: '{name_to_finalize}'") # REMOVED
        # print(f"DEBUG ({context_msg}):   Block was valid during line reading: {block_was_valid_flag}") # REMOVED
        # print(f"DEBUG ({context_msg}):   N determined for block: {N_for_block}") # REMOVED
        # print(f"DEBUG ({context_msg}):   Number of grid lines collected: {len(lines_to_finalize)}") # REMOVED
        
        if name_to_finalize and block_was_valid_flag and lines_to_finalize:
            if N_for_block > 0 and \
               len(lines_to_finalize) == N_for_block and \
               all(len(r_str) == N_for_block for r_str in lines_to_finalize):
                grid = []
                valid_puzzle_chars = True
                for r_idx_str, r_str in enumerate(lines_to_finalize):
                    row = []
                    for c_idx_char, char_val in enumerate(r_str):
                        if '1' <= char_val <= str(N_for_block): row.append(int(char_val))
                        elif char_val in '.0': row.append(0)
                        else: 
                            # print(f"DEBUG ({context_msg}): Invalid char '{char_val}' at {r_idx_str+1},{c_idx_char+1} in '{name_to_finalize}'.") # REMOVED
                            valid_puzzle_chars = False; break
                    if not valid_puzzle_chars: break
                    grid.append(row)
                
                if valid_puzzle_chars:
                    puzzles.append((name_to_finalize, grid, N_for_block))
                    # print(f"DEBUG ({context_msg}): Successfully parsed and added puzzle: '{name_to_finalize}' ({N_for_block}x{N_for_block})") # REMOVED
                else:
                    print(f"Warning (parser) ({context_msg}): Puzzle '{name_to_finalize}' had invalid characters. Skipping.") # Kept Warning
            else:
                print(f"Warning (parser) ({context_msg}): Puzzle '{name_to_finalize}' FAILED DIMENSION CHECK. Expected {N_for_block}x{N_for_block}. Actual lines: {len(lines_to_finalize)}. Skipping.") # Kept Warning
        elif name_to_finalize: 
            if not block_was_valid_flag:
                # print(f"DEBUG ({context_msg}): Puzzle block '{name_to_finalize}' was marked invalid during its line processing, not adding at finalization.") # REMOVED
                pass # Silently skip if already marked invalid
            elif not lines_to_finalize:
                # print(f"DEBUG ({context_msg}): Puzzle block '{name_to_finalize}' had no grid lines collected. Not adding.") # REMOVED
                pass # Silently skip if no lines collected for a started block

    for line_idx, line in enumerate(lines):
        if not line: 
            continue 
        if line.startswith(">"):
            finalize_puzzle(current_puzzle_name, current_grid_lines, N_of_current_grid, parsing_a_valid_puzzle_block, "New Title Trigger")
            current_puzzle_name = line[1:].strip()
            current_grid_lines = [] 
            N_of_current_grid = 0 
            parsing_a_valid_puzzle_block = True 
            # print(f"DEBUG: Started new puzzle block: '{current_puzzle_name}'") # REMOVED
        
        elif current_puzzle_name and parsing_a_valid_puzzle_block:
            if not N_of_current_grid: 
                N_of_current_grid = len(line) 
                if N_of_current_grid not in [4, 9]: 
                    print(f"Warning (parser): Puzzle '{current_puzzle_name}' line '{line}' (len {len(line)}) suggests unsupported size N={N_of_current_grid}. Marking this puzzle block invalid.") # Kept Warning
                    parsing_a_valid_puzzle_block = False 
                # else:
                    # print(f"DEBUG: Determined N={N_of_current_grid} for '{current_puzzle_name}' from line: '{line}' (len {len(line)})") # REMOVED
            
            if parsing_a_valid_puzzle_block and N_of_current_grid > 0:
                if len(line) == N_of_current_grid: 
                    if len(current_grid_lines) < N_of_current_grid: 
                        current_grid_lines.append(line)
                        # print(f"DEBUG: Added line to '{current_puzzle_name}': '{line}'. Lines collected: {len(current_grid_lines)}/{N_of_current_grid}") # REMOVED
                else: 
                    print(f"Warning (parser): Puzzle '{current_puzzle_name}' has inconsistent line length within its block. Expected {N_of_current_grid}, got {len(line)} for line '{line}'. Marking this puzzle block invalid.") # Kept Warning
                    parsing_a_valid_puzzle_block = False 

    finalize_puzzle(current_puzzle_name, current_grid_lines, N_of_current_grid, parsing_a_valid_puzzle_block, "EOF Trigger")
    # print(f"DEBUG: Finished loading puzzles. Total puzzles loaded: {len(puzzles)}") # REMOVED
    return puzzles

# # SAT Solvers (Conceptually sat_solvers.py)

def apply_literal_assignment(cnf, assigned_literal):
    new_cnf = []
    neg_assigned_literal = -assigned_literal
    for clause in cnf:
        if assigned_literal in clause:
            continue 
        new_clause = [lit for lit in clause if lit != neg_assigned_literal]
        if not new_clause: return None 
        new_cnf.append(new_clause)
    return new_cnf

def find_unit_clauses(cnf):
    unit_literals = set()
    for clause in cnf:
        if len(clause) == 1:
            unit_literals.add(clause[0])
    return unit_literals

def resolve_clauses(clause1, clause2):
    resolvent_set = set()
    resolved_on_var = None
    c1_list = list(clause1) # Ensure iterability if frozensets are passed
    c2_list = list(clause2)
    for lit1 in c1_list:
        if -lit1 in c2_list:
            resolved_on_var = abs(lit1); break
    if resolved_on_var is None: return None
    for lit in c1_list:
        if abs(lit) != resolved_on_var: resolvent_set.add(lit)
    for lit in c2_list:
        if abs(lit) != resolved_on_var:
            if -lit in resolvent_set: return None 
            resolvent_set.add(lit)
    return frozenset(sorted(list(resolvent_set))) if resolvent_set else frozenset()

def solve_resolution(initial_cnf, num_vars, max_clauses_limit=10000, timeout_seconds=10):
    start_time = time.process_time()
    clauses = {frozenset(sorted(c)) for c in initial_cnf if c}
    if frozenset() in clauses: # Check if initial CNF itself contains an empty clause
        return "UNSAT", None, len(clauses), {"message": "Initial CNF contains empty clause."}
    max_clauses_generated = len(clauses)
    details = {}
    iteration = 0
    while True:
        iteration += 1
        if time.process_time() - start_time > timeout_seconds:
            details["message"] = f"Resolution timed out after {timeout_seconds}s. Iterations: {iteration}"
            details["max_clauses_at_stop"] = len(clauses)
            return "TIMEOUT", None, len(clauses), details
        if len(clauses) > max_clauses_limit:
            details["message"] = f"Resolution clause limit ({max_clauses_limit}) exceeded. Iterations: {iteration}"
            details["max_clauses_at_stop"] = len(clauses)
            return "MEM_LIMIT", None, len(clauses), details
        new_resolvents = set()
        clauses_list = list(clauses)
        for i in range(len(clauses_list)):
            for j in range(i + 1, len(clauses_list)):
                resolvent = resolve_clauses(clauses_list[i], clauses_list[j])
                if resolvent is not None:
                    if not resolvent: 
                        details["message"] = f"Empty clause derived. Iterations: {iteration}"
                        return "UNSAT", None, max(max_clauses_generated, len(clauses)), details
                    if resolvent not in clauses:
                        new_resolvents.add(resolvent)
        if not new_resolvents:
            details["message"] = f"Resolution complete, no new clauses. Iterations: {iteration}"
            return "SAT", None, max_clauses_generated, details
        clauses.update(new_resolvents)
        max_clauses_generated = max(max_clauses_generated, len(clauses))

def _dp_simplify(cnf, assignment, num_vars): 
    simplified_cnf = copy.deepcopy(cnf) 
    made_change_in_outer_loop = True
    while made_change_in_outer_loop:
        made_change_in_outer_loop = False
        # Unit Propagation
        unit_literals_in_pass = find_unit_clauses(simplified_cnf)
        if unit_literals_in_pass:
            for unit_lit in list(unit_literals_in_pass): # Iterate on copy
                var_abs = abs(unit_lit)
                if var_abs in assignment and assignment[var_abs] != (unit_lit > 0):
                    return None, assignment # Conflict
                if var_abs not in assignment:
                    assignment[var_abs] = (unit_lit > 0)
                    new_simplified_cnf = apply_literal_assignment(simplified_cnf, unit_lit)
                    if new_simplified_cnf is None: return None, assignment 
                    simplified_cnf = new_simplified_cnf
                    made_change_in_outer_loop = True 
                    if not simplified_cnf: return [], assignment # SAT by unit prop
            if made_change_in_outer_loop: continue # Restart simplification if UP made changes
        # Pure Literal Rule (Example, can be refined)
        # This basic version does one pass. A full DP might iterate UP and Pure Literal together.
        all_literals_in_formula = set(l for cl in simplified_cnf for l in cl)
        vars_in_formula = {abs(l) for l in all_literals_in_formula}
        for var in list(vars_in_formula): # Iterate on copy of vars_in_formula
            if var not in assignment:
                is_pos_pure = var in all_literals_in_formula and -var not in all_literals_in_formula
                is_neg_pure = -var in all_literals_in_formula and var not in all_literals_in_formula
                if is_pos_pure:
                    assignment[var] = True
                    new_simplified_cnf = apply_literal_assignment(simplified_cnf, var)
                    if new_simplified_cnf is None: return None, assignment # Should not happen with pure lit
                    simplified_cnf = new_simplified_cnf
                    made_change_in_outer_loop = True
                    if not simplified_cnf: return [], assignment
                elif is_neg_pure:
                    assignment[var] = False
                    new_simplified_cnf = apply_literal_assignment(simplified_cnf, -var)
                    if new_simplified_cnf is None: return None, assignment
                    simplified_cnf = new_simplified_cnf
                    made_change_in_outer_loop = True
                    if not simplified_cnf: return [], assignment
            if made_change_in_outer_loop: break # Restart simplification if Pure Lit made changes
        if made_change_in_outer_loop: continue # Go back to unit prop
    return simplified_cnf, assignment

def solve_dp(initial_cnf, num_vars, max_clauses_limit=10000, timeout_seconds=20): 
    start_time = time.process_time()
    clauses = {frozenset(sorted(c)) for c in initial_cnf if c}
    if frozenset() in clauses:
        return "UNSAT", None, len(clauses), {"message": "Initial CNF contains empty clause."}
        
    max_clauses_generated = len(clauses)
    assignment = {}
    details = {}

    simplified_cnf_list, current_assignment = _dp_simplify([list(c) for c in clauses], assignment, num_vars)
    
    if simplified_cnf_list is None: 
        details["message"] = "UNSAT after initial simplification."
        return "UNSAT", None, max_clauses_generated, details
    if not simplified_cnf_list: 
        details["message"] = "SAT after initial simplification (all clauses removed)."
        final_model = {v: current_assignment.get(v, False) for v in range(1, num_vars + 1)}
        return "SAT", final_model, max_clauses_generated, details
    
    clauses = {frozenset(sorted(c)) for c in simplified_cnf_list}
    
    # Determine vars to eliminate based on remaining clauses and unassigned vars
    vars_in_remaining_clauses = set(abs(l) for c_set in clauses for l in c_set)
    vars_to_eliminate = sorted(list(vars_in_remaining_clauses - set(current_assignment.keys())))

    for var_to_elim in vars_to_eliminate:
        if time.process_time() - start_time > timeout_seconds:
            details["message"] = "DP timed out."
            details["max_clauses_at_stop"] = len(clauses)
            return "TIMEOUT", None, len(clauses), details
        if len(clauses) > max_clauses_limit:
            details["message"] = f"DP clause limit ({max_clauses_limit}) exceeded."
            details["max_clauses_at_stop"] = len(clauses)
            return "MEM_LIMIT", None, len(clauses), details

        clauses_with_var = {c for c in clauses if var_to_elim in c or -var_to_elim in c}
        clauses_without_var = clauses - clauses_with_var
        if not clauses_with_var: continue

        new_resolvents_for_var = set()
        pos_clauses = {c for c in clauses_with_var if var_to_elim in c}
        neg_clauses = {c for c in clauses_with_var if -var_to_elim in c}
        for c1 in pos_clauses:
            for c2 in neg_clauses:
                resolvent = resolve_clauses(c1, c2)
                if resolvent is not None:
                    if not resolvent: 
                        details["message"] = "Empty clause from DP variable elimination."
                        return "UNSAT", None, max(max_clauses_generated, len(clauses)), details
                    new_resolvents_for_var.add(resolvent)
        
        clauses = clauses_without_var.union(new_resolvents_for_var)
        max_clauses_generated = max(max_clauses_generated, len(clauses))
        
        # Re-simplify after elimination
        simplified_cnf_list, current_assignment = _dp_simplify([list(c) for c in clauses], current_assignment, num_vars)
        if simplified_cnf_list is None: 
            details["message"] = "UNSAT after simplification post-elimination."
            return "UNSAT", None, max_clauses_generated, details
        if not simplified_cnf_list: 
            details["message"] = "SAT after simplification post-elimination (all clauses removed)."
            final_model = {v: current_assignment.get(v, False) for v in range(1, num_vars + 1)}
            return "SAT", final_model, max_clauses_generated, details
        clauses = {frozenset(sorted(c)) for c in simplified_cnf_list}

    # Final check
    if not clauses:
        details["message"] = "DP complete, all clauses eliminated/satisfied."
    else: # Check if remaining clauses are satisfied by the assignment
        for c_set in clauses:
            satisfied = False
            for lit in c_set:
                var = abs(lit)
                if var in current_assignment:
                    if (lit > 0 and current_assignment[var]) or (lit < 0 and not current_assignment[var]):
                        satisfied = True; break
                else: # Unassigned var in a clause means it's not trivially satisfied by assigned vars
                    satisfied = False; break 
            if not satisfied:
                details["message"] = "DP complete, but remaining clauses are not satisfied."
                return "UNSAT", None, max_clauses_generated, details
        details["message"] = "DP complete, all remaining clauses satisfied."
        
    final_model = {v: current_assignment.get(v, False) for v in range(1, num_vars + 1)}
    return "SAT", final_model, max_clauses_generated, details

_dpll_decision_count = 0 
def _select_variable_dpll(cnf, assignment, num_vars, heuristic_strategy):
    unassigned_vars = [v for v in range(1, num_vars + 1) if v not in assignment]
    if not unassigned_vars: return None
    if heuristic_strategy == "first_unassigned": return sorted(unassigned_vars)[0]
    elif heuristic_strategy == "random_unassigned": return random.choice(unassigned_vars)
    elif heuristic_strategy == "most_frequent_simple":
        counts = Counter()
        for clause in cnf: 
            for literal in clause:
                var = abs(literal)
                if var in unassigned_vars: counts[var] += 1
        if not counts: return sorted(unassigned_vars)[0] 
        return counts.most_common(1)[0][0]
    return sorted(unassigned_vars)[0]

def _unit_propagate_dpll(cnf, assignment): 
    # print(f"DEBUG_UP: Enter _unit_propagate_dpll. Initial assignment size: {len(assignment)}") # REMOVED
    local_cnf = [list(c) for c in cnf] 
    iteration = 0
    while True:
        iteration += 1
        # print(f"DEBUG_UP: Iteration {iteration}. Clauses: {sum(1 for c in local_cnf if c is not None)}. Assignment size: {len(assignment)}") # REMOVED
        unit_literal_found_in_pass = None
        found_unit_clause_index = -1
        for i, clause_orig in enumerate(local_cnf):
            if clause_orig is None: continue
            unresolved_literals_in_clause = []
            is_satisfied = False
            for lit in clause_orig:
                var = abs(lit)
                if var in assignment: 
                    if (lit > 0 and assignment[var]) or \
                       (lit < 0 and not assignment[var]): 
                        is_satisfied = True; break 
                else: 
                    unresolved_literals_in_clause.append(lit)
            if is_satisfied:
                if local_cnf[i] is not None: local_cnf[i] = None 
                continue
            if len(unresolved_literals_in_clause) == 1:
                unit_literal_found_in_pass = unresolved_literals_in_clause[0]
                found_unit_clause_index = i
                # print(f"DEBUG_UP: Found unit literal {unit_literal_found_in_pass} in clause {clause_orig} (index {i}) -> unresolved part {unresolved_literals_in_clause}") # REMOVED
                break 
            elif len(unresolved_literals_in_clause) == 0 and not is_satisfied : 
                # print(f"DEBUG_UP: CONFLICT! Clause {clause_orig} (index {i}) became empty and is not satisfied. Unresolved: {unresolved_literals_in_clause}. Assignment: {assignment}") # REMOVED
                return None 
        if unit_literal_found_in_pass is None:
            break 
        var_to_assign = abs(unit_literal_found_in_pass)
        val_to_assign = (unit_literal_found_in_pass > 0)
        if var_to_assign in assignment:
            if assignment[var_to_assign] != val_to_assign:
                # print(f"DEBUG_UP: CONFLICT! Unit literal {unit_literal_found_in_pass} contradicts existing assignment for var {var_to_assign} (is {assignment[var_to_assign]}).") # REMOVED
                return None 
            else:
                if found_unit_clause_index != -1 : local_cnf[found_unit_clause_index] = None 
                continue 
        # print(f"DEBUG_UP: Assigning var {var_to_assign} = {val_to_assign} due to unit literal {unit_literal_found_in_pass}.") # REMOVED
        assignment[var_to_assign] = val_to_assign
        if found_unit_clause_index != -1: local_cnf[found_unit_clause_index] = None
    final_simplified_cnf = []
    for i, clause_orig in enumerate(local_cnf):
        if clause_orig is None: continue
        current_clause_literals = []
        is_clause_satisfied = False
        for literal in clause_orig:
            var = abs(literal)
            if var in assignment: 
                if (literal > 0 and assignment[var]) or \
                   (literal < 0 and not assignment[var]):
                    is_clause_satisfied = True; break 
            else: 
                current_clause_literals.append(literal)
        if is_clause_satisfied: continue
        if not current_clause_literals: 
            # print(f"DEBUG_UP: CONFLICT! Final check: Clause {clause_orig} (index {i}) became empty and is not satisfied by final assignment. Assignment: {assignment}") # REMOVED
            return None 
        final_simplified_cnf.append(current_clause_literals)
    return final_simplified_cnf

def _dpll_recursive(cnf, assignment, num_vars, heuristic_strategy, current_depth, timeout_at_time):
    global _dpll_decision_count
    if time.process_time() > timeout_at_time: raise TimeoutError("DPLL Timeout")
    
    # Pass a deepcopy of assignment to _unit_propagate_dpll if it modifies it AND you need original for false branch
    # Or, handle assignment copying/restoration carefully around recursive calls
    # Current _unit_propagate_dpll modifies assignment in-place.
    # We need to ensure assignment state is correctly managed for backtracking.
    
    temp_assignment = copy.deepcopy(assignment) # Work with a copy for this level's UP
    propagated_cnf = _unit_propagate_dpll(cnf, temp_assignment) # cnf is already a copy from parent

    if propagated_cnf is None: return False, {} # Conflict found by UP
    if not propagated_cnf: return True, temp_assignment # SAT found by UP

    # If UP succeeded and formula not empty, now we make a decision
    # _dpll_decision_count += 1 # This should be per *branching decision*, not per recursive call start
    
    chosen_var = _select_variable_dpll(propagated_cnf, temp_assignment, num_vars, heuristic_strategy)
    if chosen_var is None: # All variables assigned by UP, and propagated_cnf not empty (should be empty if SAT)
        return True # Should have been caught by `not propagated_cnf` if truly SAT

    _dpll_decision_count += 1 # Count this as one decision point

    # Try True branch
    assignment_true_branch = copy.deepcopy(temp_assignment)
    assignment_true_branch[chosen_var] = True
    # print(f"DEBUG_DPLL: Depth {current_depth}, Trying Var {chosen_var}=True, Parent Assignment Size: {len(temp_assignment)}")
    is_sat_true, model_true = _dpll_recursive(propagated_cnf, assignment_true_branch, num_vars, heuristic_strategy, current_depth + 1, timeout_at_time)
    if is_sat_true: return True, model_true
    
    # Try False branch
    # _dpll_decision_count += 1 # Not a new decision, but the other path of the same decision var
    assignment_false_branch = copy.deepcopy(temp_assignment) # Start from assignment *before* true branch
    assignment_false_branch[chosen_var] = False
    # print(f"DEBUG_DPLL: Depth {current_depth}, Trying Var {chosen_var}=False, Parent Assignment Size: {len(temp_assignment)}")
    is_sat_false, model_false = _dpll_recursive(propagated_cnf, assignment_false_branch, num_vars, heuristic_strategy, current_depth + 1, timeout_at_time)
    if is_sat_false: return True, model_false
    
    return False, {}


def solve_dpll(initial_cnf, num_vars, heuristic_strategy="first_unassigned", timeout_seconds=60):
    global _dpll_decision_count
    _dpll_decision_count = 0 
    start_time = time.process_time()
    timeout_at_time = start_time + timeout_seconds
    # The main assignment dictionary that will be built up by the successful recursive path
    final_assignment_solution = {} 
    details = {}
    try:
        # Pass a mutable copy of clauses. Assignment is built by successful recursion.
        is_sat, model_from_recursion = _dpll_recursive(
            [list(c) for c in initial_cnf], 
            {}, # Start with an empty assignment for the top-level call's scope
            num_vars, 
            heuristic_strategy, 0, timeout_at_time
        )
        if is_sat:
            final_assignment_solution = model_from_recursion # Capture the successful assignment
            
    except TimeoutError:
        details["message"] = f"DPLL timed out after {timeout_seconds}s."
        return "TIMEOUT", None, _dpll_decision_count, details
    except RecursionError:
        details["message"] = "DPLL exceeded recursion depth."
        return "ERROR", None, _dpll_decision_count, details

    if is_sat:
        # Ensure all variables have an assignment if problem is SAT for full model
        # This defaults unassigned (not in model_from_recursion) to False.
        # For Sudoku, a complete solution should assign all relevant variables.
        complete_model = {v: final_assignment_solution.get(v, False) for v in range(1, num_vars + 1)}
        details["message"] = "DPLL found a satisfying assignment."
        return "SAT", complete_model, _dpll_decision_count, details
    else:
        details["message"] = "DPLL determined unsatisfiability."
        return "UNSAT", None, _dpll_decision_count, details


# Main Comparison (Conceptually main_comparison.py)


if __name__ == "__main__":
    puzzle_file_to_run = "sudoku_puzzles.txt" 

    # Create a sample sudoku_puzzles.txt if it doesn't exist for testing
    # This part can be commented out once you have your actual puzzle files
    if not os.path.exists(puzzle_file_to_run):
        print(f"Sample puzzle file '{puzzle_file_to_run}' not found. Creating one for demonstration.")
        with open(puzzle_file_to_run, 'w') as f:
            f.write(">4x4_A_VeryEasy_Sample\n")
            f.write("1234\n3412\n2143\n432.\n\n")
            f.write(">9x9_G_Easy_Standard_Corrected_Sample\n")
            f.write("53..7....\n6..195...\n.98....6.\n8...6...3\n4..8.3..1\n7...2...6\n.6....28.\n..419..5.\n....8..79\n")
        print(f"Created '{puzzle_file_to_run}'. Please replace with your actual puzzle file(s).")

    loaded_puzzles = load_sudoku_puzzles_from_file(puzzle_file_to_run)

    if not loaded_puzzles:
        print(f"No puzzles loaded from '{puzzle_file_to_run}'. Exiting.")
        exit()
    
    solvers_config_common = [ 
        ("Resolution", solve_resolution, None), 
        ("DP", solve_dp, None),
        ("DPLL (first_unassigned)", solve_dpll, "first_unassigned"),
        ("DPLL (random_unassigned)", solve_dpll, "random_unassigned"),
        ("DPLL (most_frequent_simple)", solve_dpll, "most_frequent_simple"),
    ]

    for puzzle_name, puzzle_grid, N_size in loaded_puzzles:
        print(f"\n\n{'='*60}")
        print(f"--- Solving: {puzzle_name} ({N_size}x{N_size}) ---")
        print("Initial Grid:")
        print(format_sudoku_grid_to_string(puzzle_grid, N_size))
        print(f"{'='*60}\n")

        print("Encoding Sudoku to CNF...")
        encode_start_time = time.process_time()
        cnf = generate_sudoku_cnf(puzzle_grid, N_size)
        encode_end_time = time.process_time()

        num_vars_puzzle = N_size * N_size * N_size
        num_clauses_initial = len(cnf)
        print(f"Encoding Time: {encode_end_time - encode_start_time:.4f}s")
        print(f"CNF Generated: {num_vars_puzzle} variables, {num_clauses_initial} clauses.")
        print("-" * 60)

        for solver_display_name_template, solver_func, heuristic in solvers_config_common:
            full_solver_name = solver_display_name_template
            if heuristic: 
                full_solver_name = f"{solver_display_name_template} [{heuristic}]"
            
            print(f"\n>>> Running: {full_solver_name} on {puzzle_name}")
            
            current_cnf_for_solver = copy.deepcopy(cnf) 

            solver_params = {}
            if N_size == 4:
                if solver_func == solve_resolution: solver_params = {"max_clauses_limit": 30000, "timeout_seconds": 30}
                elif solver_func == solve_dp: solver_params = {"max_clauses_limit": 30000, "timeout_seconds": 45}
                elif solver_func == solve_dpll: solver_params = {"timeout_seconds": 60}
            elif N_size == 9:
                if solver_func == solve_resolution: solver_params = {"max_clauses_limit": num_clauses_initial + 10000, "timeout_seconds": 10} 
                elif solver_func == solve_dp: solver_params = {"max_clauses_limit": num_clauses_initial + 10000, "timeout_seconds": 20} 
                elif solver_func == solve_dpll: 
                    # Slightly longer timeout for very hard known puzzles if needed for DPLL
                    if "Hard" in puzzle_name or "Minimal" in puzzle_name or "Blank" in puzzle_name :
                         solver_params = {"timeout_seconds": 600} 
                    else:
                         solver_params = {"timeout_seconds": 300}
            else: 
                 solver_params = {"timeout_seconds": 10}

            tracemalloc.start() 
            solve_start_time = time.process_time()
            status, model, aux_metric, details_dict = "ERROR", None, 0, {"message":"Solver not configured or crashed early."}

            try:
                if solver_func == solve_dpll:
                    status, model, decisions, details_dict = solver_func(
                        current_cnf_for_solver, num_vars_puzzle, 
                        heuristic_strategy=heuristic, 
                        timeout_seconds=solver_params.get("timeout_seconds", 60))
                    aux_metric = decisions 
                elif solver_func in [solve_resolution, solve_dp]:
                     status, model, max_clauses, details_dict = solver_func(
                         current_cnf_for_solver, num_vars_puzzle,
                         max_clauses_limit=solver_params.get("max_clauses_limit", 10000),
                         timeout_seconds=solver_params.get("timeout_seconds", 10))
                     aux_metric = max_clauses 
                else:
                    print(f"Error: Unknown solver function configured: {solver_func}")
                    if tracemalloc.is_tracing(): tracemalloc.stop() 
                    continue
            except Exception as e:
                print(f"  ERROR during {full_solver_name} execution: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for unexpected errors
                status = "CRASH"
                details_dict = {"message": str(e)}
            
            solve_end_time = time.process_time()
            # Ensure tracemalloc is stopped even if solver failed early but not via critical exception
            peak_mem_solve = 0
            if tracemalloc.is_tracing():
                _, peak_mem_solve = tracemalloc.get_traced_memory()
                tracemalloc.stop() 

            print(f"  Status: {status}")
            if details_dict and details_dict.get("message"):
                 print(f"  Details: {details_dict['message']}")
            print(f"  Time Taken: {solve_end_time - solve_start_time:.4f}s")
            print(f"  Peak Memory (Python objects by tracemalloc): {peak_mem_solve / 1024**2:.4f} MB")
            
            if solver_func == solve_dpll:
                print(f"  DPLL Decisions: {aux_metric}")
            elif solver_func in [solve_resolution, solve_dp]:
                 actual_clauses_at_stop = aux_metric
                 if status in ['TIMEOUT', 'MEM_LIMIT'] and details_dict.get('max_clauses_at_stop') is not None:
                     actual_clauses_at_stop = details_dict['max_clauses_at_stop']
                 print(f"  Max Clauses Generated/Reached: {actual_clauses_at_stop}")

            if status == "SAT" and model:
                print("  Solution Found:")
                decoded_grid = decode_sudoku_solution(model, num_vars_puzzle, N_size)
                if decoded_grid:
                    print(format_sudoku_grid_to_string(decoded_grid, N_size))
                else:
                    print("  Error decoding solution or model was incomplete.")
            elif status == "SAT" and not model: 
                 print(f"  Solution: SAT (Model not explicitly constructed by this version of {solver_display_name_template})")
            
            print("-" * 40) 
        print(f"--- Finished all solvers for: {puzzle_name} ---")