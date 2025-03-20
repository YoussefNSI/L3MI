section .text

; f32 dp_fpu(f32 *x, f32 *y, u32 size) ;
dp_fpu:
    push ebp
    mov ebp, esp
    push esi
    push edi

    mov esi, [ebp+8]
    mov edi, [ebp+12]
    mov edx, [ebp+16]

    fldz        ; float sum = 0
    xor ecx, ecx ; i = 0
.for
    cmp ecx, edx
    jge .endfor

    faddp st1, st0

    inc ecx
    jmp .for
.endfor
    pop edi
    pop esi
    mov esp, ebp
    pop ebp
    ret