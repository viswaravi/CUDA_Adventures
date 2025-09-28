import numpy as np

def analyze_reduction_conflicts(block_size, num_banks=32, warp_size=32):
    """
    Analyzes bank conflicts for the reduction kernel pattern:
    sdata[2*s*tid] += sdata[2*s*tid + s];
    
    Args:
        block_size (int): The number of threads in the block (e.g., 256).
        num_banks (int): The number of shared memory banks (default 32).
        warp_size (int): The number of threads in a warp (default 32).
    """
    
    # We only need to analyze the threads in the first warp (0 to 31)
    # as conflicts are exclusively per-warp.
    tids_to_analyze = np.arange(warp_size)
    
    print(f"--- Bank Conflict Analysis for Block Size: {block_size}, Banks: {num_banks} ---")
    
    s = 1
    while s < block_size:
        # A dictionary to count how many threads access each bank in this stage 's'
        bank_access_counts = {}
        
        # Data for the sample table (first 4 active threads)
        table_data = []
        
        # The number of threads actively participating in this reduction stage
        # The max tid that is active is where 2*s*tid + s < block_size
        max_active_tid = (block_size - s - 1) // (2 * s)
        
        for tid in tids_to_analyze:
            # Check the active thread condition from the kernel:
            # if (2 * s * tid + s < block_size)
            if tid <= max_active_tid:
                # --- Index A: sdata[2*s*tid]
                index_A = 2 * s * tid
                bank_A = index_A % num_banks
                
                # --- Index B: sdata[2*s*tid + s]
                index_B = index_A + s
                bank_B = index_B % num_banks
                
                # Record access counts
                bank_access_counts[bank_A] = bank_access_counts.get(bank_A, 0) + 1
                bank_access_counts[bank_B] = bank_access_counts.get(bank_B, 0) + 1
                
                # Collect data for the table
                if True:
                    table_data.append({
                        's': s, 'tid': tid, 
                        'idxA': index_A, 'bankA': bank_A,
                        'idxB': index_B, 'bankB': bank_B
                    })

        # The conflict factor is the maximum number of accesses to a single bank
        conflict_factor = max(bank_access_counts.values()) if bank_access_counts else 1
        
        # Calculate the theoretical speedup loss: Total number of memory transactions / 2 
        # (since each thread does 2 accesses: read and write in the same line)
        # Here we simplify the conflict factor as a direct metric.
        
        print(f"\nStage s = {s}:")
        print(f"  Maximum Active tid in warp: {min(max_active_tid, warp_size - 1)}")
        print(f"  Max Bank Conflict Factor: {conflict_factor}x (i.e., {conflict_factor} accesses serialized)")
        
        # Print table sample
        print(f"  Sample Accesses (tid 0 to {len(table_data) - 1}):")
        print(f"  {'tid':<4} | {'Index A':<7} | {'Bank A':<6} | {'Index B':<7} | {'Bank B':<6}")
        print("---" * 12)
        for data in table_data:
            print(f"  {data['tid']:<4} | {data['idxA']:<7} | {data['bankA']:<6} | {data['idxB']:<7} | {data['bankB']:<6}")
            
        # Move to the next stage
        s *= 2
        
    print("\n--- Analysis Complete ---")

# Run the analysis with a common block size
# analyze_reduction_conflicts(block_size=64, num_banks=32)

def reduceCPU(length):
    array = [1] * length
    total_sum = 0.0
    for i in range(length):
        total_sum += array[i]
    return total_sum

print(reduceCPU(1024 * 1024 * 512))
