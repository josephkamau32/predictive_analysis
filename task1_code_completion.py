"""
AI for Software Engineering - Task 1: AI-Powered Code Completion
Author: JOSEPH KAMAU
Date: November 2025
Environment: Google Colab

This script compares AI-suggested code completion with manual implementation
for sorting a list of dictionaries by a specific key.
"""

# ============================================================================
# METHOD 1: AI-SUGGESTED IMPLEMENTATION (GitHub Copilot Style)
# ============================================================================

def sort_dict_list_ai(dict_list, key, reverse=False):
    """
    AI-suggested approach to sort a list of dictionaries by a specific key.
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Dictionary key to sort by
        reverse (bool): If True, sort in descending order. Default is False
        
    Returns:
        list: Sorted list of dictionaries
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    try:
        # Using sorted() with lambda - clean and pythonic
        return sorted(dict_list, key=lambda x: x.get(key, float('inf')), reverse=reverse)
    except Exception as e:
        print(f"Error sorting: {e}")
        return dict_list


# ============================================================================
# METHOD 2: MANUAL IMPLEMENTATION (Traditional Approach)
# ============================================================================

def sort_dict_list_manual(dict_list, key, reverse=False):
    """
    Manual implementation using bubble sort algorithm for educational comparison.
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Dictionary key to sort by
        reverse (bool): If True, sort in descending order. Default is False
        
    Returns:
        list: Sorted list of dictionaries
        
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    # Create a copy to avoid modifying the original list
    result = dict_list.copy()
    n = len(result)
    
    # Bubble sort implementation
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            # Handle missing keys gracefully
            val1 = result[j].get(key, float('inf'))
            val2 = result[j + 1].get(key, float('inf'))
            
            # Compare based on sort order
            if reverse:
                condition = val1 < val2
            else:
                condition = val1 > val2
            
            # Swap if needed
            if condition:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        
        # Optimization: break if no swaps occurred
        if not swapped:
            break
    
    return result


# ============================================================================
# METHOD 3: OPTIMIZED MANUAL IMPLEMENTATION (Using Built-in Sort)
# ============================================================================

def sort_dict_list_optimized(dict_list, key, reverse=False):
    """
    Optimized manual approach using list.sort() with custom comparison.
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Dictionary key to sort by
        reverse (bool): If True, sort in descending order. Default is False
        
    Returns:
        list: Sorted list of dictionaries
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    result = dict_list.copy()
    
    # Using in-place sort with key function
    result.sort(key=lambda x: x.get(key, float('inf')), reverse=reverse)
    
    return result


# ============================================================================
# PERFORMANCE TESTING & COMPARISON
# ============================================================================

import time
import random

# Create test data
def generate_test_data(size=1000):
    """Generate sample data for testing."""
    return [
        {
            'id': i,
            'name': f'Item_{i}',
            'priority': random.randint(1, 10),
            'score': random.uniform(0, 100),
            'timestamp': random.randint(1000000, 9999999)
        }
        for i in range(size)
    ]

# Performance testing function
def test_performance(func, data, key, iterations=5):
    """
    Test the performance of a sorting function.
    
    Args:
        func: Function to test
        data: Test data
        key: Key to sort by
        iterations: Number of test iterations
        
    Returns:
        float: Average execution time in seconds
    """
    times = []
    
    for _ in range(iterations):
        test_data = data.copy()  # Fresh copy for each iteration
        start_time = time.time()
        func(test_data, key)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return sum(times) / len(times)


# ============================================================================
# DEMONSTRATION & COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AI-POWERED CODE COMPLETION: COMPARATIVE ANALYSIS")
    print("="*70)
    
    # Small dataset demonstration
    print("\n1. FUNCTIONALITY TEST (Small Dataset)")
    print("-" * 70)
    
    sample_data = [
        {'name': 'Alice', 'age': 30, 'score': 85.5},
        {'name': 'Bob', 'age': 25, 'score': 92.0},
        {'name': 'Charlie', 'age': 35, 'score': 78.3},
        {'name': 'Diana', 'age': 28, 'score': 88.7},
    ]
    
    print(f"\nOriginal Data:\n{sample_data}")
    
    # Test AI-suggested method
    sorted_ai = sort_dict_list_ai(sample_data, 'score', reverse=True)
    print(f"\nAI-Suggested (sorted by score, descending):\n{sorted_ai}")
    
    # Test manual method
    sorted_manual = sort_dict_list_manual(sample_data, 'score', reverse=True)
    print(f"\nManual Implementation (sorted by score, descending):\n{sorted_manual}")
    
    # Test optimized method
    sorted_optimized = sort_dict_list_optimized(sample_data, 'score', reverse=True)
    print(f"\nOptimized Manual (sorted by score, descending):\n{sorted_optimized}")
    
    # Verify all methods produce the same result
    print(f"\nResults Match: {sorted_ai == sorted_manual == sorted_optimized}")
    
    # Large dataset performance comparison
    print("\n\n2. PERFORMANCE COMPARISON (Large Dataset)")
    print("-" * 70)
    
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} records:")
        large_data = generate_test_data(size)
        
        # Test AI-suggested implementation
        time_ai = test_performance(sort_dict_list_ai, large_data, 'priority')
        print(f"  AI-Suggested:       {time_ai*1000:.4f} ms")
        
        # Test manual implementation (skip for large datasets due to O(n²))
        if size <= 500:
            time_manual = test_performance(sort_dict_list_manual, large_data, 'priority')
            print(f"  Manual (Bubble):    {time_manual*1000:.4f} ms")
            print(f"  Speed Difference:   {time_manual/time_ai:.2f}x slower")
        else:
            print(f"  Manual (Bubble):    Skipped (too slow for large datasets)")
        
        # Test optimized implementation
        time_optimized = test_performance(sort_dict_list_optimized, large_data, 'priority')
        print(f"  Optimized Manual:   {time_optimized*1000:.4f} ms")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Key findings
    print("\n📝 KEY FINDINGS:")
    print("✓ AI-suggested code uses built-in sorted() - O(n log n) complexity")
    print("✓ Manual bubble sort - O(n²) complexity, significantly slower")
    print("✓ Optimized manual using list.sort() - O(n log n), comparable to AI")
    print("✓ AI code is more concise, readable, and production-ready")
    print("✓ Manual implementation useful for educational purposes only")