g++ -o hello_world_nasm.exe hello_world_nasm.o -m32global main
extern printf

section .data
    message:    db "hello world!", 10, 0
    
section .text

main:
    push ebp
    mov ebp, esp

    push dword message
    call printf
    add esp, 4

    xor eax, eax

    mov esp, ebp
    pop ebp
    ret

