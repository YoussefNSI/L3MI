global main
extern printf
extern atoi

section .data
    MAX EQU 60

    tab: times MAX dd 0

    msg_int: db "fib(%d) = %d", 10, 0
    msg_flt: db "fib(%d) = %f", 10, 0

    n: dd 0

section .text

fib_int:
    push ebp
    mov ebp, esp

    mov dword [tab], 0

    mov dword [tab + 4], 1

    mov ecx, 2
.for:
    cmp ecx, MAX
    jge .endfor

    mov eax, [tab + ecx*4 -4]
    add eax, [tab + ecx*4 -8]
    mov [tab + ecx*4], eax

    inc ecx
    jmp .for
.endfor:
    mov ecx, [ebp+8]
    mov eax, [tab + ecx*4]

    mov esp, ebp
    pop ebp
    ret

fib_flt:
    ; tab [ 0 ] = 0
    fldz
    fstp dword [tab]
    ; tab [ 1 ] = 1
    fld1
    fstp dword [tab + 4]

    ; for (int i=2; i < MAX; i++)
    mov ecx, 2
.for:
    cmp ecx, [ebp + 8]
    jg .endfor

    ; tab[i] = tab[i-1] + tab[i-2]
    fld dword [tab + ecx*4 -4]
    fadd dword [tab + ecx*4 -8]
    fstp dword [tab + ecx*4]

    inc ecx
    jmp .for
.endfor:
    mov ecx, [ebp+8] ; ecx <- n
    fld dword [tab + ecx*4]

    mov esp, ebp
    pop ebp
    ret


main:
    push ebp
    mov ebp, esp

    ; n = atoi(argv[1]);
    mov eax, [ebp+12] ; eax <- &argv[0]
    push dword [eax + 4]
    call atoi
    add esp, 4
    mov [n], eax

    ;printf("fib(%d)=%d\n", n, fib(n))
    push eax
    call fib_int
    add esp, 4
    push eax
    push dword [n]
    push dword msg_int
    call printf
    add esp, 12

    ;printf("fib(%d)=%f\n", n, fib(n))
    mov eax, [n]
    push eax
    call fib_flt
    add esp, 4
    ; st0 = fib(n)
    sub esp, 8
    fstp qword [esp]

    push dword [n]
    push dword msg_flt
    call printf
    add esp, 16

    xor eax, eax
    mov esp, ebp
    pop ebp
    ret