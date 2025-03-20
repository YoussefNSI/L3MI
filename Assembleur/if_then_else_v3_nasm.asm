global main
extern printf
extern scanf

section .data
    a: dd 0
    b: dd 0

    msg_inf: db "%d < %d", 10, 0
    msg_supe: db "%d >= %d", 10, 0

    msg_scanf: db "%d%d", 0
    msg_saisie: db "a , b ?", 0

section .text

main:
    push ebp
    mov ebp, esp

    push dword msg_saisie
    call printf
    add esp, 4

    push dword b 
    push dword a
    push dword msg_scanf
    call scanf
    add esp, 12
.if:
    mov eax, [a]
    mov ecx, [b]
    cmp eax, ecx
    jge .else
.then:
    push ecx
    push eax
    push dword msg_inf
    call printf
    add  esp, 12
    jmp .endif
.else:
    push ecx
    push eax
    push dword msg_supe
    call printf
    add  esp, 12
.endif:

    xor eax, eax ; return EXIT_SUCCESS
    mov esp, ebp
    pop ebp
    ret 0