#include <vector>
#include <algorithm>
#include <unordered_map>

int getMinSwaps(std::vector<int> arr) {
    int n = arr.size();
    
    // Extract and sort non-zero elements
    std::vector<int> nonZeros;
    for (int x : arr) {
        if (x != 0) {
            nonZeros.push_back(x);
        }
    }
    
    int m = nonZeros.size();
    std::vector<int> sorted = nonZeros;
    std::sort(sorted.begin(), sorted.end());
    
    // Count zeros in first m positions
    int zerosInPrefix = 0;
    for (int i = 0; i < m; i++) {
        if (arr[i] == 0) {
            zerosInPrefix++;
        }
    }
    
    // Count swaps needed to sort non-zeros in first m positions
    // Extract non-zeros from first m positions
    std::vector<int> nonZerosInPrefix;
    for (int i = 0; i < m; i++) {
        if (arr[i] != 0) {
            nonZerosInPrefix.push_back(arr[i]);
        }
    }
    
    // Count inversions/swaps to match with sorted
    int sortSwaps = 0;
    for (int i = 0; i < (int)nonZerosInPrefix.size(); i++) {
        if (nonZerosInPrefix[i] != sorted[i]) {
            sortSwaps++;
        }
    }
    
    return zerosInPrefix + sortSwaps;
}

int main() {
    int n;
    cin >> n;
    
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    
    cout << getMinSwaps(arr) << endl;
    
    return 0;
}