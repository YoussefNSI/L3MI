<?php
require_once __DIR__ . "/ADigraph.php";

/**
 * Computes the array $paths of dipaths going from origin node $origin to destination node $destination in digraph $digraph.
 * If $simple is true, returns the array of simple dipaths going from $origin to $destination.
 * The function is recursive and implements depth-first search (DFS) to traverse the digraph.
 * Both $path and $paths must be set to an empty array on the first call.
 * 
 * @param ADigraph $digraph The digraph on which to perform DFS
 * @param int $origin The id of the origin node.
 * @param int $destination The id of the destination node.
 * @param array[] &$path to keep track of the current path being constructed
 * @param array[] &$paths to store all dipaths computed during the search.
 * @param bool $simple Whether to extract simple dipaths only (true) or any dipath (false).
 *
 */
function DFS( ADigraph $digraph, int $origin, int $destination, array &$path, array &$paths, bool $simple = false ) : void
    {
    if ( $origin === $destination && $path != [] ) {
        $paths[] = $path;
        return;
    }
    $arcs = $digraph->getOutgoingArcs( $origin );
    if ( empty( $arcs ) ) {
        return;
    }
    // discard outgoing arcs that already appear in path based on their id (not their nodes!)
    $pathArcIds = array_keys( $path );
    $arcs
        = array_filter(
            $arcs,
            function ($arcId) use ($pathArcIds)
                {
                return ! in_array( $arcId, $pathArcIds, true );
                },
            ARRAY_FILTER_USE_KEY
        );
    if ( empty( $arcs ) ) {
        return;
    }
    // if simple path requested
    if ( $simple && ! empty($path)) {
        // nodes in path but last
        $nodes = array_map( fn ($arc) => $arc[ 0 ], $path );
        // adding last node
        $nodes[] = end( $path )[ 1 ];
        reset( $path );
        // discard outgoing arcs that connect an existing node in path
        $arcs
            = array_filter(
                $arcs,
                function ($arc) use ($nodes)
                    {
                    return ! in_array( $arc[ 1 ], $nodes, true );
                    }
            );
        if ( empty( $arcs ) ) {
            return;
        }
    }

    foreach ( $arcs as $arcId => $arc ) {
        $path1 = $path;
        $path[ $arcId ] = $arc;
        DFS( $digraph, $arc[ 1 ], $destination, $path, $paths, $simple );
        unset( $path[ $arcId ] );
    }
    $path = $path1;
}
?>