void main() {
    int a = 0;
    int b = 1;
    int next = 0;
    
    for (;;) {
        next = a + b;
        printf(next);
        a = b;
        b = next;
        if (next > 128) break;
    }
}