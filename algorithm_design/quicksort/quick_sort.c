#include <stdio.h>
#define MAX_LENGTH 64

void quickSort(int arr[], int left, int right);
int partition(int arr[], int left, int right);
void swap(int *a, int *b);


int main() {
    int arr[MAX_LENGTH] = {0};
    int n = -1;
    scanf("%d", &n);
    for(int i=0; i<n; i++){
        scanf("%d", &arr[i]);
    }
    // print
    printf("Inputed: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    quickSort(arr, 0, n - 1);
    printf("Sorted: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}


void quickSort(int arr[], int left, int right) {
    if (left < right) {
        int pivotIndex = partition(arr, left, right);
        quickSort(arr, left, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, right);
    }
}

int partition(int arr[], int left, int right) {
    int pivot = arr[left]; // choose the left item as pivot
    int i = left + 1;
    int j = right;
    // start moving points
    while (i <= j) {
        // move the left point
        while (i <= j && arr[i] < pivot) i++;
        // move the right point
        while (i <= j && arr[j] > pivot) j--;
        // swap the value
        if (i <= j) {
            swap(&arr[i], &arr[j]);
            i++; j--;
        }
    }
    // reset the pivot
    swap(&arr[left], &arr[j]);
    return j;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
