# Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest

from functools import reduce
import numpy as np

from dwave.optimization import Model, add, logical_or, logical_and, maximum


class TestNurseScheduling(unittest.TestCase):
    def test_small(self):
        #####################################
        # Nurse scheduling problem definition
        num_days = 7
        num_nurses = 5
        num_shifts = 3
        max_shifts_in_a_row = 3
        max_minutes_per_week = 480 * 6
        min_minutes_per_week = 480 * 3

        # How many minutes each shift is (all 8 hours for now)
        minutes_per_shift_data = [480] * num_shifts
        #####################################

        # Build a model that implements this problem
        model = Model()

        minutes_per_shift = [
            model.constant(np.repeat(minutes, num_nurses))
            for minutes in minutes_per_shift_data
        ]

        bit_sets = [
            model.disjoint_bit_sets(num_nurses, num_shifts + 1)
            for _ in range(num_days)
        ]
        shifts = [a for _, a in bit_sets]

        def smart_reduce(f, items):
            assert len(items) > 1
            return reduce(f, items[1:], items[0])

        is_working_day = [
            smart_reduce(logical_and, day_shifts[:-1])
            for day_shifts in shifts
        ]

        zero = model.constant(0)

        # Constraints that enforce max shifts in a row
        for day in range(num_days - max_shifts_in_a_row):
            working_more_than_max_shifts = smart_reduce(
                logical_and,
                is_working_day[day:day + max_shifts_in_a_row + 1]
            )
            for nurse in range(num_nurses):
                model.add_constraint(working_more_than_max_shifts[nurse] == zero)

        # Constraints that enforce min two shifts in a row
        for day in range(num_days - 3):
            not_working_min_shifts = logical_and(
                logical_and(shifts[day][-1], is_working_day[day + 1]),
                shifts[day + 2][-1]
            )
            for nurse in range(num_nurses):
                model.add_constraint(not_working_min_shifts[nurse] == zero)

        # Total minutes worked
        total_minutes_worked = None
        for shift in range(num_shifts):
            total_shifts_worked = smart_reduce(
                lambda x, y: x + y,
                [shifts[day][shift] for day in range(num_days)]
            )
            minutes_worked = total_shifts_worked * minutes_per_shift[shift]
            total_minutes_worked = (total_minutes_worked + minutes_worked
                                    if total_minutes_worked is not None
                                    else minutes_worked)

        min_minutes = model.constant(min_minutes_per_week)
        max_minutes = model.constant(max_minutes_per_week)
        for nurse in range(num_nurses):
            model.add_constraint(total_shifts_worked[nurse] >= min_minutes)
            model.add_constraint(total_shifts_worked[nurse] <= max_minutes)

        model.lock()

        with model.to_file() as f:
            recreated_model = Model.from_file(f)

        self.assertEqual(recreated_model.num_nodes(), model.num_nodes())
