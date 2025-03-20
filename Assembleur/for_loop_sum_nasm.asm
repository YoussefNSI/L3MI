global main
extern printf

section .data
    msg_sum: db "sum=%d", 10, 0

section .text

main:
    push    ebp
    mov     ebp, esp
   
    xor     eax, eax        ; sum = 0
   
    mov     ecx, 1          ; i = 1
.for_i:
    cmp     ecx, 11
    jge     .endfor_i
   
        add     eax, ecx    ; sum += i
       
    inc     ecx             ; ++i
    jmp     .for_i
.endfor_i: 
   
    push    eax
    push    dword msg_sum
    call    printf
    add     esp, 8
       
    xor     eax, eax        ; 0 of return 0
   
    mov     esp, ebp
    pop     ebp
    ret
 
 