#  (C) Copyright 2010-2020 Enthought, Inc., Austin, TX
#  All rights reserved.

import numpy as np
from types import MethodType
from numbers import Number

from force_bdss.api import (
    FixedMCOParameter,
    RangedMCOParameter,
    RangedVectorMCOParameter,
    ListedMCOParameter,
    CategoricalMCOParameter,
)

import nevergrad as ng


def get_attribute(ob, target_attributes):
    """ Get the value of an attribute matching a target.

    Parameters
    ----------
    ob: Any
        The object to search.
    target_attributes: set of str
        The target attributes.

    Return
    ------
    Any
        The value of the attribute or None if no match found.
    """
    # get the intersection of the object's attributes
    # with the set of targets.
    match = target_attributes.intersection(ob.__dir__())
    if match:
        try:
            # get one of the matching attributes
            attribute = getattr(ob, match.pop())
            # get the attribute's value
            # (return value if the attribute is a method)
            if isinstance(attribute, MethodType):
                return attribute()
            else:
                return attribute
        except AttributeError:
            return None

    return None


def duck_type_param(param):
    """ Duck-typing translation of any object
    into an nevergrad parameter object.

    Parameters
    ----------
    param: Any
        The object to be translated.

    Return
    ------
    ng.p
        A nevergrad parameter object.

    Notes
    -----
    Typing is done by matching the object's attributes against
    possible names (and types) that might suggest a particular nevergrad
    parameter object, in order of precedence:
        unordered set (ng.p.Choice)
        ordered set (ng.p.TransitionChoice)
        scalar (ng.p.Scalar)
        array (ng.p.Array)
    """

    # unordered set?
    v = get_attribute(param, {'choices', 'categories', 'set', 'values'})
    if isinstance(v, list):
        return ng.p.Choice(
            choices=v,
            deterministic=False
        )

    # ordered set?
    v = get_attribute(param, {'levels'})
    if isinstance(v, list):
        return ng.p.TransitionChoice(
            choices=v,
            transitions=[1.0, 1.0]
        )

    # scalar, vector or array ?
    v = get_attribute(
        param, {'initial_value', 'value', 'init', 'data', 'x0'})
    if isinstance(v, Number):
        return ng.p.Scalar(
            init=v,
            mutable_sigma=False
        )
    elif isinstance(v, list):
        return ng.p.Array(
            init=np.array(np.array(v)),
            mutable_sigma=True
        )
    elif isinstance(v, np.ndarray):
        return ng.p.Array(
            init=v,
            mutable_sigma=True
        )

    # otherwise
    return ng.p.Constant(value='null')


def translate_mco_to_ng(params):
    r""" Translate from an MCO parameter specification
    to a Nevergrad parameter specification (Instrumentation).

    Parameters
    ----------
    params: list of MCOParameter
        The MCO parameter specification.

    Return
    ------
    Instrumentation
        Nevergrad instrumentation object.

    Notes
    -----
    The Instrumentation object can be created from a list and/or
    a dict of nevergrad parameter types. The keys of the dict are the
    names of the parameters.
    These list/dict is passed to the Instrumentation __init__ as
    \*vargs/\*kwargs and they set the attributes called args/kwargs.
    The result of a nevergrad optimization is itself a Instrumentation object.

    As MCO objective functions (_score(), etc) take their parameters as a
    single list argument, rather than kwargs (i.e. parameters are indexed
    by position) we must create the Instrumentation by \*vargs. Also against
    using \*kwargs, is that MCO parameter names may not be unique (there is
    nothing to enforce this).

    We could do with making many of the paramterization arguments attributes
    of the optimizer.
    """

    instru = []
    for p in params:

        if isinstance(p, FixedMCOParameter):
            ng_param = ng.p.Constant(
                value=p.value,
            )
        elif isinstance(p, RangedVectorMCOParameter):
            ng_param = ng.p.Array(
                init=np.array(p.initial_value),
                mutable_sigma=True
            )
            ng_param.set_bounds(
                lower=np.array(p.lower_bound),
                upper=np.array(p.upper_bound),
                method="arctan"
            )
        elif isinstance(p, RangedMCOParameter):
            ng_param = ng.p.Scalar(
                init=p.initial_value,
                lower=p.lower_bound,
                upper=p.upper_bound,
                mutable_sigma=False
            )
        elif isinstance(p, ListedMCOParameter):
            ng_param = ng.p.TransitionChoice(
                choices=p.levels,
                transitions=[1.0, 1.0]
            )
        elif isinstance(p, CategoricalMCOParameter):
            ng_param = ng.p.Choice(
                choices=p.categories,
                deterministic=False
            )
        else:
            # duck-typing for non-standard.
            ng_param = duck_type_param(p)

        # add parameter to list
        instru.append(ng_param)

    # create Instrumentation object with *vargs
    return ng.p.Instrumentation(*instru)


def translate_ng_to_mco(ng_params):
    """ Translate a list of nevergrad parameter values
    to a list of MCO parameter values.

    Parameters
    ----------
    ng_params: list of Any (but usually float, ndarray or string)
        Parameter values in the nevergrad form

    Return
    ------
    mco_values: list of Any (but usually float, list or string)
        Parameter values in the MCO form

    Notes
    -----
    These are mostly the same, except for RangedVectorMCOParameter to
    ng.p.Array conversion: the value of the former is a list, whereas
    the value of the latter is a numpy array
    """

    mco_values = []
    for p in ng_params:
        if isinstance(p, np.ndarray):
            mco_values.append(p.tolist())
        else:
            mco_values.append(p)
        # ...what about any non-standard MCOParameter types?
        # (see translate_mco_to_ng(), above)

    return mco_values
