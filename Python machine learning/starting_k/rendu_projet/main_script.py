if __name__ == "__main__":
    
    # Il faut lancer le programme ici pour éviter les erreurs de multiprocessing sur Windows
    
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    print("Exécution du programme principal")
    
    from challenge_new import main 
    main()
 